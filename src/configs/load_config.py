"""
Load the config file
"""

import os
import yaml

# Get the directory where this file is located
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_CONFIG_DIR, "default.yaml")


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = _DEFAULT_CONFIG
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config