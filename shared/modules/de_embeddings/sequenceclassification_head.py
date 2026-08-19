class TensorTransformerForSequenceClassification(nn.Module):
    """
    A complete sequence classification model using the Tensor Transformer Encoder.
    """
    def __init__(self, num_layers, d, p, h, d_ff, vocab_size, T_max, num_classes,
                 pe_strategy="linear", activation="relu", norm_first=False):
        super().__init__()
        self.encoder = TensorTransformerEncoder(
            num_layers=num_layers, d=d, p=p, h=h, d_ff=d_ff,
            vocab_size=vocab_size, T_max=T_max, pe_strategy=pe_strategy,
            activation=activation, norm_first=norm_first
        )
        # Sequence classification classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(d, num_classes)
        )

    def forward(self, input_ids):
        # 1. Run through the tensorized encoder stack: (B, T, d)
        hidden_states = self.encoder(input_ids)

        # 2. Mean pooling over the sequence tokens: (B, d)
        pooled = hidden_states.mean(dim=1)

        # 3. Linear classification projections: (B, num_classes)
        return self.classifier(pooled)