
from experiments.SMRL.models.SMRL_for_seq_class import SMRL_Model_for_Sequence_Classification
from shared.baseline.models.std import Std
from pathlib import Path

def model_loader(config):
    if config["model"]["type"] == "SMRL":
        return SMRL_Model_for_Sequence_Classification(config=config)
    elif config["model"]["type"] == "Std":
        return Std(config=config)
    else:
        raise ValueError(f"Invalid model type: {config['model']['type']}")
    

def resolve_tags_name(config):
    return [     
        'SMRL',
        config["data"]["dataset_name"]
        # f"init.normal_var0.01"
        # f"orthogonality_penalty",
        # f"orthogonality_penalty_{config['model']['orth_lambda']}",
    ]

def resolve_project_name(config):
    return "SMRL_learnable_z"

def resolve_run_name(config):
    #return f'BaselineReviewed_Clas_data_{config["data"]["dataset_name"]}_bias_off'
    return f'SMRL_Clas_data_{config["data"]["dataset_name"]}_kind_{config["model"]["kind"]}'

