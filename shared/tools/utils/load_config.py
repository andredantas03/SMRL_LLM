import yaml

def load_config(config_path='experiments/SMRL/configs/classification.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config