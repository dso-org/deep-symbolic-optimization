import json
import os

from dsr.utils import safe_merge_dicts

def get_base_config(task):
    # Load base config
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "base_config.json"), encoding='utf-8') as f:
        base_config = json.load(f)

    # Load task specific config
    task_config_file = None
    if task in ["regression", "control"]:
        task_config_file = "config_dsr.json"
    elif task in ["gp"]:
        task_config_file = "config_gp.json"
    else:
        assert False, "*** ERROR: Unknown task type: {}".format(task)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), task_config_file), encoding='utf-8') as f:
        task_config = json.load(f)

    return safe_merge_dicts(base_config, task_config)