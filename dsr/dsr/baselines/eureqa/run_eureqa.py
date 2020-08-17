import os
import json
import multiprocessing

import click
import pandas as pd
import datarobot as dr


# NOTE: Mac users first run `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`


def work(arg):

    project_name, benchmark, results_path, eureqa_params, seed = arg

    print("Running {}...".format(project_name))

    # Get the project
    project = get_project(project_name=project_name,
                          benchmark=benchmark)

    # Get the base model
    base_model = get_base_model(project=project)

    # Get the custom model
    model = get_model(project=project,
                      base_model=base_model,
                      eureqa_params=eureqa_params,
                      seed=seed)

    # Get the solution
    solution = model.get_pareto_front().solutions[-1].expression
    solution = solution.split("Target = ")[-1]

    # Compute success
    success = get_success(benchmark, solution)

    # # Delete the project
    # project.delete()

    # Append results
    df = pd.DataFrame({
        "name" : [benchmark],
        "seed" : [seed],
        "project_name" : [project_name],
        "project_id" : [project.id],
        "base_model_id" : [base_model.id],
        "model_id" : [model.id],
        "solution" : [solution],
        "success" : [success]
        })
    
    return df


def get_success(benchmark, solution):
    """Determine whether the solution is symbolically correct."""

    return "TBD"


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


def get_project(project_name, benchmark):
    """Create a project with the given name and benchmark."""

    df = get_dataset(benchmark)
    project = dr.Project.start(project_name=project_name,
                               sourcedata=df,
                               autopilot_on=False,
                               target='y')

    return project


def get_base_model(project):
    """Create the base model."""

    # Find the Eureqa symbolic regression algorithm
    bp = [bp for bp in project.get_blueprints() if "Eureqa" in bp.model_type and "Instant" in bp.model_type][0]

    # Train the base model (required before adjusting parameters)
    model_job_id = project.train(bp, sample_pct=100.0)
    job = dr.ModelJob.get(model_job_id=model_job_id, project_id=project.id)
    model = job.get_result_when_complete()
        
    return model


def get_model(project, base_model, eureqa_params, seed):
    """Create the custom model."""

    # Set custom parameters
    tune = base_model.start_advanced_tuning_session()
    for parameter_name, value in eureqa_params.items():
        tune.set_parameter(parameter_name=parameter_name, value=value)

    # Set the seed
    tune.set_parameter(parameter_name="random_seed", value=seed)

    # Train the custom model
    job = tune.run()
    model = job.get_result_when_complete()

    return model


@click.command()
@click.argument("results_path", type=str, default="results.csv") # Path to existing results CSV, to checkpoint runs.
@click.option("--config", type=str, default="config.json", help="Path to Eureqa JSON configuration.")
@click.option("--mc", type=int, default=100, help="Number of seeds to run.")
@click.option("--num_workers", type=int, default=8, help="Number of workers.")
@click.option("--prefix", type=str, default=None, help="Project name prefix.")
@click.option("--seed_shift", type=int, default=0, help="Starting seed value.")
def main(results_path, config, mc, num_workers, prefix, seed_shift):
    """Run Eureqa on Nguyen benchmarks for multiple random seeds."""

    # Load Eureqa paremeters
    eureqa_params = load_config(config)

    # Load existing results to skip over
    if os.path.isfile(results_path):
        df = pd.read_csv(results_path)
        completed_projects = df["project_name"].tolist()
        write_header = False
    else:
        write_header = True

    # Set up jobs
    args = []
    seeds = [i + seed_shift for i in range(mc)]
    n_benchmarks = 12
    for seed in seeds:
        for i in range(n_benchmarks):
            benchmark = "Nguyen-{}".format(i+1)
            project_name = '_'.join([benchmark, str(seed)])
            if prefix is not None:
                project_name = '_'.join([prefix, project_name])
            if project_name in completed_projects:
                print("Skipping project {} as it already completed.".format(project_name))
                continue
            arg = (project_name, benchmark, results_path, eureqa_params, seed)
            args.append(arg)

    # Farm out the work
    if num_workers > 1:
        pool = multiprocessing.Pool(num_workers, initializer=start_client, initargs=())
        for result in pool.imap_unordered(work, args):
            pd.DataFrame(result, index=[0]).to_csv(results_path, header=write_header, mode='a', index=False)
            write_header = False
    else:
        start_client()
        for arg in args:
            result = work(arg)
            pd.DataFrame(result, index=[0]).to_csv(results_path, header=write_header, mode='a', index=False)
            write_header = False


if __name__ == "__main__":
    main()
