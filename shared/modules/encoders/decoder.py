from shared.tools.functions.embedding_tensorizer import EmbeddingTensorizer
import torch
import torch.nn as nn
from shared.modules.positional_encoders.sliceawarepositionalencoding import SliceAwarePositionalEncoding
from shared.modules.transformers_blocks.smrl_transformer_block import SMRLTransformerBlock
from shared.tools.functions.dcttransform import DCTTransform
import lightning as L

class SMRLTransformerDecoder(L.LightningModule):
    """
    A full Stack of N SMRL Transformer Encoder Layers (Definition 5.10).
    """
    def __init__(self, num_layers, d, p, h, d_ff, vocab_size, T_max, 
                 pe_strategy="linear", activation="relu", norm_first=False, causal=False):
        super().__init__()
        self.d = d
        self.p = p
        self.h = h
        self.d_s = d // p
        self.vocab_size = vocab_size

        Z = DCTTransform.get_matrix(self.p, dtype=torch.float32, kind='dct')  # ou o kind do config
        self.register_buffer("Z", Z)

        # Word embeddings in standard 2D space
        self.token_embeddings = nn.Embedding(vocab_size, d)
        
        # Folding / unfolding utilities
        self.tensorizer = EmbeddingTensorizer(p)

        # Slice-aware positional encoding
        self.positional_encoding = SliceAwarePositionalEncoding(T_max, self.d_s, p, strategy=pe_strategy)

        # Core encoder layer stack
        self.layers = nn.ModuleList([
            SMRLTransformerBlock(d, p, h, d_ff, activation=activation, norm_first=norm_first, causal=causal)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids, mask):
        B, s = input_ids.shape
        device = input_ids.device

        # 1. Standard token embeddings: (B, s, d)
        x_emb = self.token_embeddings(input_ids)

        # 2. Reshape/fold embeddings to tensor space: (B, s, ds, p)
        X = self.tensorizer.tenp(x_emb)

        # 3. Dynamic positional encoding generation
        P = self.positional_encoding(B, s, device)

        # 4. Forward pass through N sequential tensor encoder blocks
        for layer in self.layers:
            X = layer(X, P, self.Z)

        # 5. Reconstruct standard representations: (B, s, d)
        return self.tensorizer.matp(X)
