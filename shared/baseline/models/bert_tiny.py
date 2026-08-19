import torch
import torch.nn as nn
import lightning as L

from shared.tools.functions.loss import cross_entropy_loss
from shared.tools.utils.optimizer_config import build_optimizer_and_scheduler
from shared.tools.utils.gradient_monitoring import GradientNormLoggingMixin
from shared.baseline.modules.embeddings.embeddings import Embedding
from shared.baseline.modules.norms.rmsnorm import RMSNorm
from shared.baseline.modules.transformers_blocks.bert_encoder_block import BertEncoder_Block


class BertTiny(GradientNormLoggingMixin, L.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        model_cfg = config["model"]

        self.d_model = model_cfg["hidden_size"]
        self.vocab_size = model_cfg["vocab_size"]
        self.num_layers = model_cfg["n_layer"]
        self.n_head = model_cfg["n_head"]
        self.d_ff = model_cfg["d_ff"]
        self.dropout = model_cfg["dropout"]
        self.num_classes = model_cfg["num_classes"]
        self.cls_index = model_cfg.get("cls_index", 0)

        self.embedding = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
        )

        self.transformer_blocks = torch.nn.ModuleDict(
            {
                f"block_{i}": BertEncoder_Block(
                    d_model=self.d_model,
                    n_head=self.n_head,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                )
                for i in range(self.num_layers)
            }
        )

        self.norm = RMSNorm(d_model=self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        self.reset_classifier()

    @torch.no_grad()
    def reset_classifier(self):
        std = self.d_model ** -0.5
        nn.init.normal_(self.classifier.weight, mean=0.0, std=std)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for i in range(self.num_layers):
            x = self.transformer_blocks[f"block_{i}"](x, attention_mask=attention_mask)
        x = self.norm(x)
        cls = x[:, self.cls_index]
        return self.classifier(cls)

    def _shared_step(self, batch, stage: str):
        logits = self(batch["input_ids"], attention_mask=batch.get("attention_mask"))
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
