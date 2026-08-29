import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch
from experiments.SMRL.models.SMRL_for_seq_class import SMRL_Model_for_Sequence_Classification
from experiments.SMRL.names import model_loader, resolve_kwargs_classification, resolve_project_name, resolve_run_name, resolve_tags_name
from shared.baseline.models.bert_tiny import BertTiny
from shared.data.dataset_loader import build_classification_dataloaders, load_config
import warnings
import wandb


warnings.filterwarnings("ignore")




def train(
    config,
    model_name: str,
):

    #model = BertTiny(config=config)
    model = model_loader(config=config)
    n_params = sum(p.numel() for p in model.parameters())
    #model = torch.compile(model)
    

    dataloaders = build_classification_dataloaders(
        **resolve_kwargs_classification(config)
    )

    

    wandb_logger = WandbLogger(
        project=resolve_project_name(config),
        name=resolve_run_name(config),
        job_type=model_name,
        tags=resolve_tags_name(config),
        log_model=False,
        config=config,
        save_dir=config["paths"]["save_dir"]+'/'+config["experiment"]["task"]+"/",
    )

    
    
    

    trainer_kwargs = {
        "max_epochs": config["training"]["max_epochs"],
        "devices": 1,
        "accelerator": "gpu",
        "logger": wandb_logger,
        "precision": "16-mixed",
        "gradient_clip_val": config["training"]["gradient_clip_val"],
        "limit_val_batches": config["logging"]["limit_val_batches"],
        "default_root_dir": "results/lightning",
    }
    
    trainer = L.Trainer(**trainer_kwargs)

    test_result = None
    try:
        trainer.fit(model, dataloaders["train"], dataloaders["test"])
        test_result = trainer.test(model, dataloaders["test"])
    finally:
        wandb_logger.finalize("success")
        wandb.finish()

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
    parser = argparse.ArgumentParser(description="Train a single SMRL experiment.")
    parser.add_argument(
        "--config",
        default="experiments/SMRL/configs/classification.yaml",
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
    base_seed = config.get("experiment", {}).get("seed", 42)
    L.seed_everything(base_seed)
    train(config, args.model)
