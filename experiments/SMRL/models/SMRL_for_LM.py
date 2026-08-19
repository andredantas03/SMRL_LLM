from shared.modules.de_embeddings.smrl_lm_head import SMRL_LM_Head
from shared.modules.encoders.decoder import SMRLTransformerDecoder
import lightning as L

from shared.tools.functions.loss import cross_entropy_loss
from shared.tools.utils.gradient_monitoring import GradientNormLoggingMixin
from shared.tools.utils.optimizer_config import build_optimizer_and_scheduler

class SMRL_Model_for_Language_Modeling(GradientNormLoggingMixin,L.LightningModule):
    """
    A complete language modeling model using the SMRL Transformer Encoder.
    """
    def __init__(self,config,
                 pe_strategy="linear", activation="relu", norm_first=False):
        super().__init__()
        pe_strategy="linear"
        activation="relu"
        norm_first=False
        num_layers = config["model"]["n_layer"]
        d = config["model"]["hidden_size"]
        p = config["model"]["p"]
        h = config["model"]["n_head"]
        d_ff = config["model"]["d_ff"]
        vocab_size = config["model"]["vocab_size"]
        T_max = config["model"]["max_seq_length"]
        self.config = config


        self.decoder = SMRLTransformerDecoder(
            num_layers=num_layers, d=d, p=p, h=h, d_ff=d_ff,
            vocab_size=vocab_size, T_max=T_max, pe_strategy=pe_strategy,
            activation=activation, norm_first=norm_first, causal=True
        )
        # Language Model Head
        self.lm_head = SMRL_LM_Head(d, p, vocab_size)

    def forward(self, x, mask):
        # x tem shape (batch_size sequence_length)
        hidden_states = self.decoder(x, mask)
        logits = self.lm_head(hidden_states)
        return logits

    
    def training_step(self, batch, batch_idx=None):          
        input_ids=batch["input_ids"]
        labels=batch["labels"]
        mask=batch["attention_mask"]
        logits = self(input_ids, mask)
        loss = cross_entropy_loss(logits,labels)        
        self.log_loss_and_ppl(loss, "train_loss", on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        input_ids=batch["input_ids"]
        labels=batch["labels"]
        mask=batch["attention_mask"]       
        logits = self(input_ids, mask)
        loss = cross_entropy_loss(logits,labels)  
        self.log_loss_and_ppl(loss, "val_loss", on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids=batch["input_ids"]
        labels=batch["labels"]       
        mask=batch["attention_mask"]
        logits = self(input_ids, mask)
        loss = cross_entropy_loss(logits,labels) 
        self.log_loss_and_ppl(loss, "test_loss", on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return build_optimizer_and_scheduler(self, self.config)
    
    
    
    