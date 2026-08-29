
from experiments.SMRL.models.SMRL_for_seq_class import SMRL_Model_for_Sequence_Classification
from shared.baseline.models.bert_tiny import BertTiny
from pathlib import Path

def model_loader(config):
    if config["model"]["type"] == "SMRL":
        return SMRL_Model_for_Sequence_Classification(config=config)
    elif config["model"]["type"] == "BertTiny":
        return BertTiny(config=config)
    else:
        raise ValueError(f"Invalid model type: {config['model']['type']}")
    

def resolve_tags_name(config):
    return [     
        'Baseline_Reviewed',
        config["data"]["dataset_name"]
        # f"init.normal_var0.01"
        # f"orthogonality_penalty",
        # f"orthogonality_penalty_{config['model']['orth_lambda']}",
    ]

def resolve_project_name(config):
    return "SMRL_learnable_z"

def resolve_run_name(config):
    return f'BaselineReviewed_Clas_data_{config["data"]["dataset_name"]}_bias_off'
    #return f'SMRL_Clas_data_{config["data_classification"]["dataset_name"]}\
        # _kind_{config["model"]["kind"]}\
        # _ortl_{config["model"]["orth_lambda"]}\
        #     _initnorm_var0.01'

def resolve_kwargs_classification(config):
    task = config["experiment"]["task"]
    dataset_name = config["data"]["dataset_name"]
    processed_dataset_path = Path(f'shared/data/processed/{dataset_name}/30000_sennrich/').as_posix()

    
    
    return {
        "processed_dataset_path": processed_dataset_path,
        "batch_size": config["training"]["batch_size"],
        "eval_batch_size": config["training"]["eval_batch_size"],
        "num_workers": config["data"]["num_workers"],
        "context_length": config["model"]["max_seq_length"],
        "pad_id": config["data"]["pad_id"],
    }