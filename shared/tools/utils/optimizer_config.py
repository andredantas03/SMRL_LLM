# tools/lightning/optimizer_config.py
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from shared.tools.functions.learning_rate import learning_rate_schedule


def build_optimizer_and_scheduler(self, config: dict):
    train = config["training"]
    decay, no_decay = [], []
    for name, param in self.named_parameters():
        if name.endswith("learnable_z.z"):
            no_decay.append(param)
        else:
            decay.append(param)
    if no_decay:
        param_groups = [
            {"params": decay, "weight_decay": train["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    else:
        param_groups = [
            {"params": decay, "weight_decay": train["weight_decay"]},
        ]
    optimizer = AdamW(
        param_groups,
        lr=train["lr"],            # 3e-4
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