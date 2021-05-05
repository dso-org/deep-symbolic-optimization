import json
import os

def get_base_config():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "base_config.json"), encoding='utf-8') as f:
        base_config = json.load(f)
    return base_config