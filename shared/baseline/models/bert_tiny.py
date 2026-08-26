import torch
import torch.nn as nn
import lightning as L

from shared.tools.functions.loss import cross_entropy_loss
from shared.tools.utils.optimizer_config import build_optimizer_and_scheduler
from shared.tools.utils.gradient_monitoring import GradientNormLoggingMixin
from shared.baseline.modules.transformers_blocks.bert_encoder_block import BertEncoder_Block
from shared.baseline.modules.positional_encoders.sinusoidal import SinusoidalPositionalEncoding


class BertTiny(GradientNormLoggingMixin, L.LightningModule):
    """Paper Std: vanilla Transformer encoder, Post-LN, sinusoidal PE, mean-pool + linear."""

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        model_cfg = config["model"]

        self.d_model = model_cfg["hidden_size"]
        self.vocab_size = model_cfg["vocab_size"]
        self.num_layers = model_cfg["n_layer"]
        self.n_head = model_cfg["H"]
        self.d_ff = model_cfg["d_ff"]
        self.dropout = model_cfg["dropout"]
        self.num_classes = config["data_classification"]["num_classes"]
        pad_id = config["data_classification"].get("pad_id", 0)

        self.token_embeddings = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=pad_id
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            model_cfg["max_seq_length"], self.d_model
        )
        self.emb_dropout = nn.Dropout(self.dropout)

        self.transformer_blocks = torch.nn.ModuleDict(
            {
                f"block_{i}": BertEncoder_Block(
                    d_model=self.d_model,
                    n_head=self.n_head,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation="gelu",
                )
                for i in range(self.num_layers)
            }
        )

        self.classifier = nn.Linear(self.d_model, self.num_classes)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        if self.token_embeddings.padding_idx is not None:
            self.token_embeddings.weight[self.token_embeddings.padding_idx].zero_()
        nn.init.normal_(self.classifier.weight, mean=0.0, std=self.d_model ** -0.5)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb_dropout(self.positional_encoding(self.token_embeddings(input_ids)))
        for i in range(self.num_layers):
            x = self.transformer_blocks[f"block_{i}"](x, attention_mask=attention_mask)
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)

    def _shared_step(self, batch, stage: str):
        logits = self(batch["input_ids"], attention_mask=batch["attention_mask"])
        labels = batch["labels"]
        loss = cross_entropy_loss(logits, labels)

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        if stage == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        elif stage == "val":
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx=None):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx=None):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(self, self.config)
