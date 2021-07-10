import os
import json
import multiprocessing

import click
import numpy as np
import pandas as pd
import datarobot as dr
from sympy.utilities.lambdify import lambdify


MAX_WAIT = 3600
THRESHOLD = 1e-10


def work(arg):

    try:

        benchmark, results_path, eureqa_params, seed = arg

        print("Running {} with seed {}...".format(benchmark, seed))

        # Get the project
        project = get_project(project_name=benchmark)

        # Get the base model
        base_model = get_base_model(project=project)

        # Get the custom model
        model = get_model(project=project,
                          base_model=base_model,
                          eureqa_params=eureqa_params,
                          seed=seed)

        # For noisy datasets, evaluate all solutions along the pareto front.
        # If success, use that. Otherwise, use the most complex solution.
        if "_n" in benchmark:
            for solution in model.get_pareto_front().solutions:
                solution = solution.expression.split("Target = ")[-1]
                nmse_test, nmse_test_noiseless, success = evaluate(benchmark, solution)
                if success:
                    break

        # Otherwise, evaluate the best solution
        else:            
            solution = model.get_pareto_front().solutions[-1].expression
            solution = solution.split("Target = ")[-1]
            nmse_test, nmse_test_noiseless, success = evaluate(benchmark, solution)

        # Append results
        df = pd.DataFrame({
            "benchmark" : [benchmark],
            "seed" : [seed],
            "project_id" : [project.id],
            "base_model_id" : [base_model.id],
            "model_id" : [model.id],
            "solution" : [solution],
            "nmse_test" : [nmse_test],
            "nmse_test_noiseless" : [nmse_test_noiseless],
            "success" : [success]
            })

    except Exception as e:
        print("Hit '{}' exception for {} on seed {}!".format(e, benchmark, seed))
        df = None
        
    return df


def evaluate(benchmark, solution):
    """Evaluate the solution against the benchmark."""

    # Get test data
    root = os.path.dirname(__file__)
    path = os.path.join(root, "../../task/regression/data/{}_test.csv".format(benchmark))
    df_test = pd.read_csv(path, header=None)
    X = df_test.values[:, :-1].T # X values are the same for noisy/noiseless
    y_test = df_test.values[:, -1]
    var_y_test = np.var(y_test)

    # Get noiseless test data
    if "_n" in benchmark and "_d" in benchmark:
        noise_str = benchmark.split('_')[1]
        benchmark_noiseless = benchmark.replace(noise_str, "n0.00")
        path = os.path.join(root, "../../task/regression/data/{}_test.csv".format(benchmark_noiseless))
        df_test_noiseless = pd.read_csv(path, header=None)
        y_test_noiseless = df_test_noiseless.values[:, -1]
        var_y_test_noiseless = np.var(y_test_noiseless)
    else:
        y_test_noiseless = y_test
        var_y_test_noiseless = var_y_test

    # Parse solution as sympy expression
    inputs = ["x{}".format(i+1) for i in range(X.shape[0])]
    solution = solution.replace("^", "**")
    f_hat = lambdify(inputs, solution, "numpy")    

    # Compare against test data
    y_hat = f_hat(*X)
    nmse_test = np.mean((y_test - y_hat)**2) / var_y_test
    nmse_test_noiseless = np.mean((y_test_noiseless - y_hat)**2) / var_y_test_noiseless
    success = nmse_test_noiseless < THRESHOLD

    return nmse_test, nmse_test_noiseless, success


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


def get_project(project_name):
    """Get the project, or create one if it doesn't exist."""

    # If the project exists, return it.
    for project in dr.Project.list():
        if project.project_name == project_name:
            return project

    # Otherwise, create the project.
    df = get_dataset(project_name)
    project = dr.Project.start(project_name=project_name,
                               sourcedata=df,
                               autopilot_on=False,
                               target='y')

    # Unlock holdout dataset (required to access all training data)
    project.unlock_holdout()

    return project


def get_base_model(project):
    """Get the base model, or create one if it doesn't exist."""

    # If the base model exists in this project, return it.
    models = project.get_models()

    if len(models) > 0:
        model = models[0]

    else:
        # Find the Eureqa symbolic regression algorithm
        bp = [bp for bp in project.get_blueprints() if "Eureqa" in bp.model_type and "Instant" in bp.model_type][0]

        # Train the base model (required before adjusting parameters)
        model_job_id = project.train(bp, sample_pct=100.0)
        job = dr.ModelJob.get(model_job_id=model_job_id, project_id=project.id)
        model = job.get_result_when_complete(max_wait=MAX_WAIT)
        
    return model


def get_model(project, base_model, eureqa_params, seed):
    """Get the model, or create one if it doesn't exist."""

    # Set custom parameters
    tune = base_model.start_advanced_tuning_session()
    for parameter_name, value in eureqa_params.items():
        tune.set_parameter(parameter_name=parameter_name, value=value)

    # Set the seed
    tune.set_parameter(parameter_name="random_seed", value=seed)

    # Train the custom model.
    # The model may have already been run, which causes an error. This can
    # happen when the connection is lost, so the model completes on the server
    # but is not recorded on the client. When this happens, search for the model
    # with identical parameters and return it.
    try:
        job = tune.run()
        model = job.get_result_when_complete(max_wait=MAX_WAIT)
    except dr.errors.JobAlreadyRequested:
        print("Job was already requested! Searching for existing model...")
        models = project.get_models()
        eureqa_params_copy = eureqa_params.copy()
        eureqa_params_copy["random_seed"] = seed
        for model in models:
            parameters = model.start_advanced_tuning_session().get_parameters()["tuning_parameters"]
            parameters = {p["parameter_name"] : p["current_value"] for p in parameters}
            if parameters == eureqa_params_copy:
                print("Found existing model!", model.id)
                return model
        assert False, "Job was alredy run but could not find existing model."

    return model


# NOTE: Mac users first run `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
# To run main experiments on Nguyen benchmarks: `python run_eureqa.py`
# To run main experiments on Constant benchmarks: `python run_eureqa.py --benchmark_set=Constant`
# To run noise experiments: `python run_eureqa.py results_noise.csv --mc=10 --seed_shift=1000 --sweep`
# To run test experiment: `python run_eureqa.py results_test.csv --num_workers=2 --mc=3 --seed_shift=123 --benchmark_set=Test`
@click.command()
@click.argument("results_path", type=str, default="results.csv") # Path to existing results CSV, to checkpoint runs.
@click.option("--config", type=str, default="config.json", help="Path to Eureqa JSON configuration.")
@click.option("--mc", type=int, default=100, help="Number of seeds to run.")
@click.option("--num_workers", type=int, default=8, help="Number of workers.")
@click.option("--seed_shift", type=int, default=0, help="Starting seed value.")
@click.option("--sweep", is_flag=True, help="Run noise and dataset size experiments.")
@click.option("--benchmark_set", type=click.Choice(["Nguyen", "Constant", "Test", "Custom"]), default="Nguyen", help="Choice of benchmark set.")
def main(results_path, config, mc, num_workers, seed_shift, sweep, benchmark_set):
    """Run Eureqa on benchmarks for multiple random seeds."""

    # Load Eureqa paremeters
    eureqa_params = load_config(config)
    if benchmark_set == "Constant":
        eureqa_params["building_block__constant"] = 1

    # Load existing results to skip over
    if os.path.isfile(results_path):
        df = pd.read_csv(results_path)
        write_header = False
    else:
        df = None
        write_header = True

    # Define the work
    args = []
    seeds = [i + seed_shift for i in range(mc)]
    n_benchmarks = {
        "Nguyen" : 12,
        "Constant" : 4,
        "Test" : 1,
        "Custom" : 22
    }[benchmark_set]
    for seed in seeds:
        for i in range(n_benchmarks):

            # # Hack to only run hard-coded Custom benchmarks
            # custom_include = [1, 2, 3, 4, 6, 11, 12, 13, 16]
            # if benchmark_set == "Custom" and i+1 not in custom_include:
            #     continue

            benchmarks = []
            if sweep: # Add all combinations of noise and dataset size multipliers
                assert benchmark_set == "Nguyen", "Noise sweep only supported for Nguyen benchmarks."
                noises = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
                dataset_size_multipliers = [1, 10]
                for n in noises:
                    for d in dataset_size_multipliers:
                        benchmark = "{}-{}_n{:.2f}_d{:.0f}".format(benchmark_set, i+1, n, d)
                        benchmarks.append(benchmark)
            else: # Just add the noiseless benchmark
                benchmark = "{}-{}".format(benchmark_set, i+1)
                benchmarks.append(benchmark)
            
            for benchmark in benchmarks:
                if df is not None and len(df.query("benchmark=='{}' and seed=={}".format(benchmark, seed))) > 0:
                    print("Skipping benchmark {} with seed {} as it already completed.".format(benchmark, seed))
                    continue
                arg = (benchmark, results_path, eureqa_params, seed)
                args.append(arg)

    # Farm out the work
    if num_workers > 1:
        pool = multiprocessing.Pool(num_workers, initializer=start_client, initargs=())
        for result in pool.imap_unordered(work, args):
            if result is not None:
                pd.DataFrame(result, index=[0]).to_csv(results_path, header=write_header, mode='a', index=False)
                write_header = False
    else:
        start_client()
        for arg in args:
            result = work(arg)
            pd.DataFrame(result, index=[0]).to_csv(results_path, header=write_header, mode='a', index=False)
            write_header = False

    # If running in test mode, delete the models
    if benchmark_set == "Test":
        if num_workers > 1:
            start_client()
        project = get_project(benchmark)
        for model in project.get_models():
            print("Deleting test model {}.".format(model.id))
            model.delete()


if __name__ == "__main__":
    main()
