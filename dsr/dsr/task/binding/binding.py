import os
import numpy as np
import pandas as pd
import torch

import dsr
from dsr.library import Library, Token
from dsr.functions import create_tokens
import dsr.constants as constants

import abag_ml.rl_environment_objects as rl_env_obj
import vaccine_advance_core.featurization.vaccine_advance_core_io as vac_io
import abag_agent_setup.expand_allowed_mutant_menu as abag_agent_setup_eamm


def make_binding_task(name, paths):
    """
    Factory function for ab/ag binding affinity rewards. 

    Parameters
    ----------
    name : str or None
        Name of AbAg study being performed.

    paths : dict
        Path to files used to run Gaussian Process-based binding environment

    Returns
    -------

    task : Task
        Dynamically created Task object whose methods contains closures.
    """

    # get master sequence
    master_seqrecord = vac_io.list_of_seqrecords_from_fasta(
        os.path.join(paths['base_path'], paths['master_seqrecord_fasta'])
    )[0]

    # load Gaussian Process data
    x = torch.load(os.path.join(paths['base_path'], paths['history_x_tensor']))
    i = torch.load(os.path.join(paths['base_path'], paths['history_i_tensor']))
    y = torch.load(os.path.join(paths['base_path'], paths['history_y_tensor']))

    env = rl_env_obj.GPModelEnvironment(
        os.path.join(paths['base_path'], paths['model_weights_pth']),
        os.path.join(paths['base_path'], paths['master_structure']),
        master_seqrecord,
        ('A', 'C'),  # TODO: check vs. master_structure
        'A',
        torch.ones((1,), dtype=torch.long),  # TODO: check if this must be an int or if it can be a torch.long
        history_studies=None,
        history_tensor_x=x,
        history_tensor_i=i,
        history_tensor_y=y,
        is_sparse=paths['model_is_sparse'],
        is_mtl=paths['model_is_mtl'],
        parallel_featurization=False,
        use_gpu=paths['use_gpu'] if 'use_gpu' in paths else True
    )

    def reward(p):
        """ Compute reward value for a given program (sequence). 

            Parameters
            ----------
            p : Program
                A program that contains a single sequence.
            
            Returns:
            ----------
            rwd : Reward value

        """
        # rwd = env.reward(''.join([t.name for t in p.traversal]))
        rwd = env.reward(''.join([t.name for t in p.assemble_sequence()]))
        rwd = rwd.item()
        
        return rwd


    def evaluate(p):
        """ Compute certain statistics of the program (sequence).

            Parameters
            ----------
            p : Program
                A program that contains a single sequence.
            
            Returns:
            ----------
            info : statistics 

        """
        info = {}
        return info

    # define amino acids as tokens
    tokens = [Token(None, aa, arity=1, complexity=1) for aa in constants.AMINO_ACIDS]

    library = Library(tokens)

    extra_info = {}

    task = dsr.task.Task(reward_function=reward,
                         evaluate=evaluate,
                         library=library,
                         stochastic=False,
                         task_type='binding',
                         extra_info=extra_info)

    return task
