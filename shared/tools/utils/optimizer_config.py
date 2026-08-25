# tools/lightning/optimizer_config.py
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from shared.tools.functions.learning_rate import learning_rate_schedule


def build_optimizer_and_scheduler(self, config: dict):
    train = config["training"]
    optimizer = AdamW(
        self.parameters(),
        lr=train["lr"],            # 3e-4
        weight_decay=train["weight_decay"],
        # paper não fixa betas; 0.9/0.999 é o default do AdamW
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=train["lr"],
        total_steps=self.trainer.estimated_stepping_batches,
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=train["lr"] / train["lr_min"],  # 3e-4 / 1e-5 = 30
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }