import os
import sys
import json

import click
import pandas as pd
import datarobot as dr


def start_client():
    """Start the DataRobot client"""

    with open("credentials.json", 'r') as f:
        credentials = json.load(f)

    # Start the DataRobot client
    print("Connecting to client...")
    dr.Client(token=credentials["token"],
              endpoint=credentials["endpoint"])


def load_config(config_path):
    """Load custom Eureqa parameters"""

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def get_dataset(name):
    """Retrieve a benchmark dataset by name."""

    root = os.path.dirname(__file__)
    path = os.path.join(root, "../../task/regression/data", name + ".csv")
    df = pd.read_csv(path, header=None)
    df.columns = ["x{}".format(i+1) for i in range(len(df.columns) - 1)] + ["y"]

    return df


def get_project(project_name, results_path):
    """Retrieve an existing project by name. If it doesn't exist, create it."""

    with open(results_path, 'r') as f:
        results = json.load(f)

    if project_name in results.keys():
        project_id = results[project_name]["project_id"]
        print("Found existing project for {}: {}".format(project_name, project_id))
        project = dr.Project.get(project_id=project_id)
    
    else:
        df = get_dataset(project_name.split('_')[0]) # TBD FIX HACK
        print("Creating new project...")
        project = dr.Project.start(project_name=project_name,
                                   sourcedata=df,
                                   autopilot_on=False,
                                   target='y')
        
        # Save the project id
        print("Created new project for {}: {}".format(project_name, project.id))
        results[project_name] = {
            "project_id" : project.id,
            "base_model_id" : None,
            "model_id" : None,
            "solution": None
            }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=3)

    return project


def get_base_model(project, results_path):
    """Retrieve the base model, or create one if it does not yet exist"""

    name = project.project_name
    with open(results_path, 'r') as f:
        results = json.load(f)
    base_model_id = results[name]["base_model_id"]

    if base_model_id is None:

        print("Creating new base model...")

        # Find the Eureqa symbolic regression algorithm
        bp = [bp for bp in project.get_blueprints() if "Eureqa" in bp.model_type and "Quick" in bp.model_type][0]

        # Train the base model (required before adjusting parameters)
        model_job_id = project.train(bp)
        job = dr.ModelJob.get(model_job_id=model_job_id, project_id=project.id)
        model = job.get_result_when_complete()

        # Save the base model id
        results[name]["base_model_id"] = model.id
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=3)

    else:
        print("Found existing base model:", base_model_id)
        model = dr.Model.get(project=project.id, model_id=base_model_id)
        
    return model


def get_model(project, eureqa_params, results_path):
    """Retrieve the custom model, or create one if it does not yet exist"""

    name = project.project_name
    seed = int(name.split('_')[-1])
    with open(results_path, 'r') as f:
        results = json.load(f)    
    model_id = results[name]["model_id"]

    if model_id is None:

        # Set custom parameters
        base_model_id = results[name]["base_model_id"]
        base_model = dr.Model.get(project=project.id, model_id=base_model_id)
        tune = base_model.start_advanced_tuning_session()
        for parameter_name, value in eureqa_params.items():
            tune.set_parameter(parameter_name=parameter_name, value=value)

        # Set the seed
        tune.set_parameter(parameter_name="random_seed", value=seed)

        # Train the custom model                
        job = tune.run()
        print("Training custom model...")
        model = job.get_result_when_complete()

        # Save the custom model
        with open(results_path, 'r') as f:
            results = json.load(f)
        results[name]["model_id"] = model.id
        results[name]["solution"] = model.get_pareto_front().solutions[-1].expression
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=3)

    else:
        print("Found existing custom model:", model_id)
        model = dr.Model.get(project=project.id, model_id=model_id)

    return model


@click.command()
@click.argument("results_path", type=str, default="results.json") # Path to existing results JSON, to 'checkpoint' runs.
@click.option("--config", type=str, default="config.json", help="Path to Eureqa JSON configuration.")
@click.option("--mc", type=int, default=10, help="Number of seeds to run.")
def main(results_path, config, mc):
    """Run Eureqa on Nguyen benchmarks for multiple random seeds."""

    # Start DataRobot client
    start_client()

    # Load Eureqa paremeters
    eureqa_params = load_config(config)

    # Generate a results JSON if it doesn't exist
    if not os.path.exists(results_path):
        with open(results_path, 'w') as f:
            json.dump({}, f, indent=3)

    # Run each benchmark in series
    for benchmark in range(12):
        for seed in range(mc):

            project_name = "Nguyen-{}_{}".format(benchmark+1, seed)

            # Get the project
            project = get_project(project_name=project_name,
                                  results_path=results_path)

            # Get the base model
            base_model = get_base_model(project=project,
                                        results_path=results_path)

            # Get the custom model
            model = get_model(project=project,
                              eureqa_params=eureqa_params,
                              results_path=results_path)


if __name__ == "__main__":
    main()
