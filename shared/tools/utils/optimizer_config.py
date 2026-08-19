# tools/lightning/optimizer_config.py
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from shared.tools.functions.learning_rate import learning_rate_schedule


def build_optimizer_and_scheduler(module, config: dict):
    train = config["training"]
    amax = train["lr"]
    amin = train.get("lr_min", amax * 0.1)
    wd = train.get("weight_decay", 0.0)
    tw = train.get("warm_up_steps", 0)
    tc = train["max_steps"]
    
    optimizer = AdamW(
        module.parameters(),
        lr=amax,              # LR base; o scheduler escala isto
        weight_decay=wd,
        betas=(0.9, 0.95),    # opcional; comum em LLMs
    )
    
    def lr_lambda(step: int) -> float:
        # LambdaLR multiplica o LR inicial por este fator
        return learning_rate_schedule(step, amax, amin, tw, tc) / amax

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",   # importante: schedule por step, não por epoch
            "frequency": 1,
        },
    }