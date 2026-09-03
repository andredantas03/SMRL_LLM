import math
from shared.tools.functions.embedding_tensorizer import EmbeddingTensorizer
import torch
import torch.nn as nn
from shared.modules.positional_encoders.sliceawarepositionalencoding import SliceAwarePositionalEncoding
from shared.modules.transformers_blocks.smrl_transformer_block import SMRLTransformerBlock
from shared.tools.functions.orthogonaltransform import OrthogonalTransform
import lightning as L
class SMRLTransformerEncoder(L.LightningModule):
    """
    A full Stack of N SMRL Transformer Encoder Layers (Definition 5.10).
    """
    def __init__(self, num_layers, d, p, h, d_ff, vocab_size, T_max, kind, activation,
                 pe_strategy="linear", norm_first=False, dropout=0.1):
        super().__init__()
        self.d = d
        self.p = p
        self.h = h
        self.d_s = d // p
        self.vocab_size = vocab_size
        self.emb_dropout = nn.Dropout(dropout)
        
        self.orthogonal_transform = OrthogonalTransform(p,kind)
        

        # Word embeddings in standard 2D space
        self.token_embeddings = nn.Embedding(vocab_size, d, padding_idx=0)
        
        # Folding / unfolding utilities
        self.tensorizer = EmbeddingTensorizer(p)

        # Slice-aware positional encoding
        self.positional_encoding = SliceAwarePositionalEncoding(T_max, self.d_s, p, strategy=pe_strategy)

        # Core encoder layer stack
        self.layers = nn.ModuleList([
            SMRLTransformerBlock(d, p, h, d_ff, activation=activation, norm_first=norm_first, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.reset_parameters()
    
    @torch.no_grad()
    def reset_parameters(self):
        pass
        # nn.init.trunc_normal_(self.token_embeddings.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

        # if self.token_embeddings.padding_idx is not None:
        #     self.token_embeddings.weight[self.token_embeddings.padding_idx].zero_()

    def forward(self, input_ids, attention_mask=None):
        B, s = input_ids.shape
        device = input_ids.device

        # 1. Standard token embeddings: (B, s, d)
        x_emb = self.token_embeddings(input_ids)*math.sqrt(self.d)

        # 2. Reshape/fold embeddings to tensor space: (B, s, ds, p)
        X = self.tensorizer.tenp(x_emb)

        # 3. Dynamic positional encoding generation
        P = self.positional_encoding(B, s, device)
        X = X + P
        X = self.emb_dropout(X)
        # 4. Forward pass through N sequential tensor encoder blocks
        Z = self.orthogonal_transform.get_matrix(dtype=torch.float32)
        
        for layer in self.layers:
            X = layer(X, None, Z, attention_mask=attention_mask)

        # 5. Reconstruct standard representations: (B, s, d)
        return self.tensorizer.matp(X)
