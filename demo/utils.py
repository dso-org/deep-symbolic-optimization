import json

from dsr.run import get_dataset
from dsr.program import Program, from_tokens

# Configure the program class from config file
def configure_program(config_filename):
    with open(config_filename, "r") as f:
        config = json.load(f)

    # Set the dataset
    config_dataset = config["dataset"]
    name = config_dataset["name"]
    dataset = get_dataset(name, config_dataset)
    Program.set_training_data(dataset)

    # Set the function set
    Program.set_library(dataset.function_set, dataset.n_input_var)
    
    # Clear the cache
    Program.clear_cache()

    # Set the reward function
    reward = config["training"]["reward"]
    reward_params = config["training"]["reward_params"]
    reward_params = reward_params if reward_params is not None else []
    Program.set_reward_function(reward, *reward_params)

    # Set the complexity penalty
    complexity = config["training"]["complexity"]
    complexity_weight = config["training"]["complexity_weight"]
    Program.set_complexity_penalty(complexity, complexity_weight)

    # Set the constant optimizer
    const_optimizer = config["training"]["const_optimizer"]
    const_params = config["training"]["const_params"]
    const_params = const_params if const_params is not None else {}
    Program.set_const_optimizer(const_optimizer, **const_params)


# Return a Program object corresponding to the ith iteration
def make_program(traversal_text, i):
    traversal = traversal_text[i]
    traversal = traversal.split(",")
    traversal = Program.convert(traversal)    
    p = from_tokens(traversal, optimize=False)
    return p

# ##### USAGE #####

# # Get training data
# X = Program.X_train
# y = Program.y_train

# # Configure the Program class
# configure_program("./data/demo.json")

# # Get the best traversals at each iteration
# with open("./data/traversals.txt", "r") as f:
#     traversal_text = f.readlines()

# # Get information from each iteration's best Program
# for i in range(len(traversal_text)):
#     p = make_program(traversal_text, i)
#     r = p.r # Reward
#     text = p.sympy_expr
#     y_hat = p.execute(Program.X_train) # Predicted values on training data
#     print("Best expression:", text)
#     print("Best reward:", r)
