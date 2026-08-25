import torch
from shared.modules.encoders.encoder import SMRLTransformerEncoder
import torch.nn as nn

from shared.tools.functions.loss import cross_entropy_loss
from shared.tools.utils.optimizer_config import build_optimizer_and_scheduler
import lightning as L

class SMRL_Model_for_Sequence_Classification(L.LightningModule):
    """
    A complete sequence classification model using the Tensor Transformer Encoder.
    """
    def __init__(self, config,
                 pe_strategy="standard", activation="gelu", norm_first=False):
        super().__init__()
        self.config = config
        self.encoder = SMRLTransformerEncoder(
            num_layers=config["model"]["n_layer"],
            d=config["model"]["hidden_size"],
            p=config["model"]["p"],
            h=config["model"]["n_head"],
            d_ff=config["model"]["d_ff"],
            vocab_size=config["model"]["vocab_size"],
            T_max=config["model"]["max_seq_length"],
            kind=config["model"]["kind"],
            pe_strategy=pe_strategy,
            activation=activation,
            norm_first=norm_first,
            dropout=config["model"]["dropout"]
        )
        # Sequence classification classifier head
        self.classifier = nn.Linear(
            config["model"]["hidden_size"],
            config["data_classification"]["num_classes"],
        )

    def forward(self, input_ids, attention_mask=None):
        hidden = self.encoder(input_ids, attention_mask=attention_mask)
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)
    
    def _shared_step(self, batch, stage: str):
        logits = self(batch["input_ids"], batch.get("attention_mask"))
        labels = batch["labels"]
        loss = cross_entropy_loss(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, "train")
    def validation_step(self, batch, batch_idx=None):
        return self._shared_step(batch, "val")
    def test_step(self, batch, batch_idx=None):
        return self._shared_step(batch, "test")
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        optimizer.step(closure=optimizer_closure)   
        
        if self.config["model"]["kind"] == "learnable":
            z = self.encoder.orthogonal_transform.learnable_z.z
            with torch.no_grad():
                zf = z.float()
                Q, R = torch.linalg.qr(zf)
                s = torch.sign(torch.diag(R))
                s = torch.where(s == 0, torch.ones_like(s), s)
                z.copy_(Q * s.unsqueeze(0))
    
    def configure_optimizers(self):
        return build_optimizer_and_scheduler(self, self.config)