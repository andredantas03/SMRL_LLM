import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from experiments.SMRL.models.SMRL_for_LM import SMRL_Model_for_Language_Modeling
from shared.baseline.models.gpt import GPT
from shared.data.dataset_loader import build_dataloaders, load_config
import warnings
import wandb


warnings.filterwarnings("ignore")




def train(
    config,
    model_name: str,
):
    base_seed = config.get("experiment", {}).get("seed", 42)
    L.seed_everything(base_seed)

    model = GPT(config=config)
    #model = SMRL_Model_for_Language_Modeling(config=config)
    n_params = sum(p.numel() for p in model.parameters())
    
    
    
    limit_val_batches = config.get("logging", {}).get("limit_val_batches", 200)
    eval_batch_size = config.get("batching", {}).get(
        "eval_batch_size", config["training"]["batch_size"]
    )
    val_sample_seed = config.get("experiment", {}).get("seed", 42)

    dataloaders = build_dataloaders(
        tokenized_dataset_path=config["data"]["tokenized_dataset_path"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        context_length=config["model"]["max_seq_length"],
        max_steps=config["training"]["max_steps"],
        eval_batch_size=eval_batch_size,
        limit_val_batches=limit_val_batches,
        val_sample_seed=val_sample_seed,
    )
    tags = [
        model_name,
        f"model_{model_name}",
        "test-5k-steps",
        config["data"]["dataset_name"],
        f"vocab_{config['model']['vocab_size']}",
        "Baseline GPT",
        f"hidden_size_{config['model']['hidden_size']}",
        f"n_layer_{config['model']['n_layer']}",
        f"p_{config['model']['p']}",
        f"max_seq_length_{config['model']['max_seq_length']}",
    ]

    wandb_logger = WandbLogger(
        project="Ortogonal matrix investigation",
        name='Baseline GPT',
        job_type=model_name,
        tags=tags,
        log_model=False,
    )

    
    
    

    trainer_kwargs = {
        "max_steps": config["training"]["max_steps"],
        "devices": 1,
        "accelerator": "gpu",
        "logger": wandb_logger,
        "precision": "bf16-mixed",
        "gradient_clip_val": config["training"].get("gradient_clip_val", 0.5),
    }
    
    trainer = L.Trainer(**trainer_kwargs)

    test_result = None
    try:
        trainer.fit(model, dataloaders["train"], dataloaders["valid"])
        test_result = trainer.test(model, dataloaders["test"])
    finally:
        wandb_logger.finalize("success")

    test_loss = None
    if test_result and isinstance(test_result[0], dict):
        test_loss = test_result[0].get("test_loss")

    return {
        "test_loss": test_loss,
        "n_params": n_params,
        "test_result": test_result,
        "global_step": trainer.global_step,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single CPToken experiment.")
    parser.add_argument(
        "--config",
        default="experiments/SMRL/configs/default.yaml",
        help="Path to base config YAML.",
    )
    parser.add_argument(
        "--model",
        default="gpt",
        choices=["gpt"],
        help="Model architecture to train.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, args.model)
