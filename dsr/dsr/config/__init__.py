import copy
import json
import os
import sys

import pandas as pd

from datetime import datetime
from pkg_resources import resource_filename

from dsr.utils import safe_merge_dicts


def get_base_config(task, method):
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

    # Load method specific config
    task_config["task"]["method"] = method
    if method in ["gp"]:
        method_file = "config_gp.json"
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), method_file), encoding='utf-8') as f:
            gp_config = json.load(f)
        task_config = safe_merge_dicts(task_config, gp_config)

    return safe_merge_dicts(base_config, task_config)

def set_benchmark_configs(config, arg_benchmark, method="dsr", output_filename=None):
    """Get all indivual benchmarks and generate their respective configs."""
    # Use benchmark name from config if not specified as command-line arg
    if len(arg_benchmark) == 0:
        assert config["task"]["name"] is not None, "Task set to 'None' in config! Use the --b argument: python dsr.run config_file.json --b your_task"
        if isinstance(config["task"]["name"], str):
            benchmarks = (config["task"]["name"],)
        elif isinstance(config["task"]["name"], list):
            benchmarks = tuple(config["task"]["name"])
    else:
        benchmarks = arg_benchmark
    original_benchmarks = list(benchmarks)

    # Get log folder naming
    if any("..." in benchmark for benchmark in original_benchmarks) \
            or len(original_benchmarks) > 1:
        log_appendix = "Mixed"
    else:
        log_appendix = original_benchmarks[0]

    #If summaries are being saved on "training", summaries must be turned on under controller
    if config["training"].get("save_summary", True):
        if not config["controller"].get("summary", False):
            print('WARNING: When config["training"]["save_summary"] is true or absent, config["controller"]["summary"] '
                  'must be true. The summary recording will be turned on.')
        config["controller"]["summary"] = True

    # set common paths
    paths = {}
    paths["log_dir"] = os.path.join(
        config["training"]["logdir"],
        "log_{}_{}".format(
            datetime.now().strftime("%Y-%m-%d-%H%M%S"),
            log_appendix))

    # Create log dir and save commandline arguments
    os.makedirs(paths["log_dir"], exist_ok=True)
    with open(os.path.join(paths["log_dir"], "cmd.out"), 'w') as f:
        print(" ".join(sys.argv), file=f)

    # Save config for complete experiment
    config["task"]["name"] = original_benchmarks
    with open(os.path.join(paths["log_dir"], "config_all.json"), 'w') as f:
        json.dump(config, f, indent=4)

    # Update config where necessary
    config["training"]["logdir"] = paths["log_dir"]

    benchmark_df = None
    if config["task"]["task_type"] == "regression":
        # Dataset needs to be created
        if isinstance(config["task"]["dataset"], dict):
            paths["root_dir"] = resource_filename("dsr.task", "regression") \
                if config["task"]["dataset"]["root"] is None \
                    else config["task"]["dataset"]["root"]
            paths["benchmark_file"] = "benchmarks.csv" \
                if config["task"]["dataset"]["benchmark_source"] is None \
                    else config["task"]["dataset"]["benchmark_source"]
            paths["tokenset_path"] = os.path.join(
                paths["root_dir"], "function_sets.csv")
            if "dataset" in config["task"] \
                    and "backup" in config["task"]["dataset"] \
                    and config["task"]["dataset"]["backup"]:
                config["task"]["dataset"]["logdir"] = paths["log_dir"]
            # load all available benchmarks
            benchmark_df = pd.read_csv(
                os.path.join(paths["root_dir"], paths["benchmark_file"]),
                index_col=None, encoding="ISO-8859-1")
            # load available token sets
            if config["task"]["function_set"] is None:
                tokenset_df = pd.read_csv(
                    paths["tokenset_path"],
                    index_col=None, encoding="ISO-8859-1")

        # Dataset already exists and will be loaded
        elif isinstance(config["task"]["dataset"], str):
            paths["dataset"] = config["task"]["dataset"]
            # load token sets
            if not isinstance(config["task"]["function_set"], list):
                # Fallback to standard set
                print('WARNING: Tokenset has not been set properly, falling back to standard tokenset: '
                      '["add","sub","mul","div","sin","cos","exp","log"]')
                config["task"]["function_set"] = ["add","sub","mul","div","sin","cos","exp","log"]
        else:
            assert False, "Unknown dataset source: {}".format(config["task"]["function_set"])

    # Helper functions
    def _set_individual_paths(benchmark):
        new_paths = copy.deepcopy(paths)
        new_paths["config_file"] = "{}_{}_config.json".format(method, benchmark)
        new_paths["config_path"] = os.path.join(
            paths["log_dir"], new_paths["config_file"])
        if output_filename is None:
            new_paths["summary_file"] = "{}_{}_summary.csv".format(method, benchmark)
            new_paths["summary_path"] = os.path.join(
                paths["log_dir"], new_paths["summary_file"])
        else:
            new_paths["summary_file"] = output_filename.split("/")[-1]
            new_paths["summary_path"] = output_filename
        return new_paths

    def _set_individual_config(benchmark):
        new_config = copy.deepcopy(config)
        new_config["task"]["name"] = benchmark
        if not isinstance(config["task"]["function_set"], list):
            tokenset_name = benchmark_df[
                benchmark_df["name"]==benchmark]["function_set"].item()
            new_config["task"]["function_set"] = tokenset_df[
                tokenset_df["name"]==tokenset_name]["function_set"].item().split(',')
        new_config["paths"] = _set_individual_paths(benchmark)
        with open(new_config["paths"]["config_path"], 'w') as f:
            json.dump(new_config, f, indent=4)
        new_config["task"].pop("method")
        new_config["task"].pop("runs")
        new_config["task"].pop("seed")
        return new_config

    # make sure we get the right benchmarks
    benchmarks = {}
    for benchmark in original_benchmarks:
        if benchmark[-3:] == "...":
            benchmark_list = list(benchmark_df['name'].loc[benchmark_df['name'].str.startswith(benchmark[:-3])])
            for each_benchmark in benchmark_list:
                benchmarks[each_benchmark] = _set_individual_config(each_benchmark)
            continue
        benchmarks[benchmark] = _set_individual_config(benchmark)
    return benchmarks

def load_config(config_template=None, method="dsr", runs=None):
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
        if "dataset" in personal_config["task"] and isinstance(personal_config["task"]["dataset"], str):
            personal_config["task"]["name"] = personal_config["task"]["dataset"].split("/")[-1][:-4]

    # Load base config
    base_config = get_base_config(task, method)
    if runs is not None:
        base_config["task"]["runs"] = int(runs)

    # Return combined configs
    return safe_merge_dicts(base_config, personal_config)