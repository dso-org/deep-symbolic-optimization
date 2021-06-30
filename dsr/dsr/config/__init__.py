import json
import os

from dsr.utils import safe_merge_dicts


def get_base_config(task, language_prior):
    # Load base config
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_common.json"), encoding='utf-8') as f:
        base_config = json.load(f)

    # Load task specific config
    task_config_file = None
    if task in ["regression", None]:
        task_config_file = "config_regression.json"
    elif task in ["control"]:
        task_config_file = "config_control.json"
    elif task in ["binding"]:
        task_config_file = "config_binding.json"
    else:
        assert False, "*** ERROR: Unknown task type: {}".format(task)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), task_config_file), encoding='utf-8') as f:
        task_config = json.load(f)

    # Load language prior config
    if language_prior:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_language.json"), encoding='utf-8') as f:
            language_config = json.load(f)
        task_config = safe_merge_dicts(task_config, language_config)

    return safe_merge_dicts(base_config, task_config)


def load_config(config_template=None, task="regression", language_prior=False):
    # Load personal config file
    personal_config = {}
    task = None
    if config_template is not None:
        # Load personalized config
        with open(config_template, encoding='utf-8') as f:
            personal_config = json.load(f)
        try:
            task = personal_config["task"]["task_type"]
        except KeyError:
            pass
        try:
            language_prior = personal_config["prior"]["language_model"]["on"]
        except KeyError:
            pass

    # Load base config
    base_config = get_base_config(task, language_prior)

    # Return combined configs
    return safe_merge_dicts(base_config, personal_config)
