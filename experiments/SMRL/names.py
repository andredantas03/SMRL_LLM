
from experiments.SMRL.models.SMRL_for_seq_class import SMRL_Model_for_Sequence_Classification
from shared.baseline.models.bert_tiny import BertTiny

def model_loader(config):
    if config["model"]["type"] == "SMRL":
        return SMRL_Model_for_Sequence_Classification(config=config)
    elif config["model"]["type"] == "BertTiny":
        return BertTiny(config=config)
    else:
        raise ValueError(f"Invalid model type: {config['model']['type']}")
    

def resolve_tags_name(config):
    return [
        config["model"]["type"],
        f"model_{config['model']['type']}",
        f"max_epochs_{config['training_classification']['max_epochs']}",
        config["data_classification"]["processed_dataset_path"]["train_docs_path"],
        f"vocab_{config['model']['vocab_size']}",
        f"hidden_size_{config['model']['hidden_size']}",
        f"n_layer_{config['model']['n_layer']}",
        f"p_{config['model']['p']}",
        f"max_seq_length_{config['model']['max_seq_length']}",
        f"kind_{config['model']['kind']}",
        f"experiment_{config['experiment']['name']}",
    ]

def resolve_project_name(config):
    return "SMRL"

def resolve_run_name(config):
    return "SMRL-"+config["model"]["kind"]