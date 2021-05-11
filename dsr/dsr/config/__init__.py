import json
import os

from dsr.utils import safe_merge_dicts

def get_base_config(task, method, language_prior):
    # Load base config
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_common.json"), encoding='utf-8') as f:
        base_config = json.load(f)

    # Load task specific config
    task_config_file = None
    if task in ["regression"]:
        task_config_file = "config_regression.json"
    elif task in ["control"]:
        task_config_file = "config_control.json"
    else:
        assert False, "*** ERROR: Unknown task type: {}".format(task)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), task_config_file), encoding='utf-8') as f:
        task_config = json.load(f)

    # Load method specific config
    task_config["task"]["method"] = method
    if method in ["gp", "gp_meld"]:
        if method in ["gp"]:
            method_file = "config_gp.json"
        else:
            method_file = "config_gp_meld.json"
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), method_file), encoding='utf-8') as f:
            gp_config = json.load(f)
        task_config = safe_merge_dicts(task_config, gp_config)

    # Load language prior config
    if language_prior:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_language.json"), encoding='utf-8') as f:
            language_config = json.load(f)
        task_config = safe_merge_dicts(task_config, language_config)

    return safe_merge_dicts(base_config, task_config)