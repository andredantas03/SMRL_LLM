from torch import Tensor,nn
import torch

from shared.baseline.tools.functions.loss import cross_entropy_loss
from shared.baseline.modules.transformers_blocks.transformer_block import Transformer_Block
from shared.baseline.modules.embeddings.embeddings import Embedding
from shared.baseline.modules.norms.rmsnorm import RMSNorm
import lightning as L

from shared.tools.utils.optimizer_config import build_optimizer_and_scheduler
from shared.tools.utils.gradient_monitoring import GradientNormLoggingMixin

class GPT(GradientNormLoggingMixin, L.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.d_model     = config['model']['hidden_size']
        self.vocab_size  = config['model']['vocab_size']
        self.num_layers  = config['model']['n_layer']
        self.max_seq_len = config['model']['max_seq_length']
        self.n_head      = config['model']['H']
        self.d_ff        = config['model']['d_ff']
        self.dropout     = config['model']['dropout']
        
        self.embedding = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)

        self.transformer_blocks = torch.nn.ModuleDict({
            f"block_{i}": Transformer_Block(
                d_model= self.d_model,
                n_head = self.n_head,
                d_ff = self.d_ff,
                dropout = self.dropout,
            )
            for i in range(self.num_layers)
        })

        self.norm = RMSNorm(d_model=self.d_model)
        self.linear = nn.Linear(in_features=self.d_model, out_features=self.vocab_size)

    def forward(self, x: Tensor):
        # x tem shape (batch_size sequence_length)
        x = self.embedding(x)  # output shape (batch_size sequence_length d_model)
        for i in range(self.num_layers):
            x = self.transformer_blocks[f"block_{i}"](x)  
            # output shape (batch_size sequence_length d_model)
        x = self.norm(x)  # output shape (batch_size sequence_length d_model)
        x = self.linear(x)  # output shape (batch_size sequence_length vocab_size)
        return x

    def training_step(self, batch, batch_idx=None):        
        input_ids=batch["input_ids"]
        labels=batch["labels"]       
        logits = self(input_ids)
        loss = cross_entropy_loss(logits,labels)
        self.log_loss_and_ppl(loss, "train_loss", on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        input_ids=batch["input_ids"]
        labels=batch["labels"]       
        logits = self(input_ids)
        loss = cross_entropy_loss(logits,labels)
        self.log_loss_and_ppl(loss, "val_loss", on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids=batch["input_ids"]
        labels=batch["labels"]       
        logits = self(input_ids)
        loss = cross_entropy_loss(logits,labels)
        self.log_loss_and_ppl(loss, "test_loss", on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return build_optimizer_and_scheduler(self, self.config)
    
