""" driver module

This module runs the program.

You can run it with
    python3 driver.py

"""

# system imports
import os
from os.path import abspath, join, dirname, isdir
import sys
from datetime import datetime as dt

# third party imports
import numpy as np
from numpy.random import default_rng
import tensorflow as tf


# local imports
import flags
import log_conf
import kfac
import neural_net as NN
import monte_carlo as MC
import quantum_mechanics as QM
from log_conf import log, log_small_horizontal_line, log_large_horizontal_line


def prepare_file_paths():
    """ x """

    try:
        if not isdir(flags.result_folder):
            os.makedirs(flags.result_folder, exist_ok=True)
    except Exception as e:
        print(
            f"Failed to create results folder at {flags.result_folder}"
            "Is there an issue with the OS and/or file system?"
        )
        raise e

    time = dt.now().strftime("%b_%d_%Hh_%Mm_%Ss")

    result_path = join(flags.result_folder, f'results_{time}')

    os.makdir(result_path)

    return result_path


def prepare_system(result_path):
    """ x """

    path = join(result_path, flags.cmdline_file)

    data = ' '.join(sys.argv[1:]) + '\n'

    # save command line arguments
    with open(path, 'w') as f:
        f.write(data)

    if flags.system_type == 'molecule':
        args = None

    elif flags.system_type == 'atom':
        args = None

    elif flags.system_type == 'atom':
        args = None

    else:
        raise ValueError(f'Incorrect system type {flags.system_type}!')

    return args


def prepare_nework(*args):
    """ x """
    kwargs = {}
    network_configuration = NN.NetworkConfiguration(*args, **kwargs)

    return network_configuration


def prepare_optimizer(*args):
    """ x """
    kwargs = {}
    optimizaiton_configuration = NN.OptimizerConfiguration(*args, **kwargs)

    return optimizaiton_configuration


def prepare_kfac(*args):
    """ x """
    kwargs = {}
    kfac_configuration = kfac.KfacConfiguration(*args, **kwargs)

    return kfac_configuration


def prepare_mcmc(*args):
    """ x """
    kwargs = {}
    mcmc_configuration = kfac.McmcConfiguration(*args, **kwargs)

    return mcmc_configuration


def main():
    """ x """
    args = ()

    if flags.deterministic:
        # set random seed to flags.deterministic_seed
        log.info('Running in deterministic mode. Performance will be reduced.')

    result_path = prepare_file_paths()
    args = prepare_system()
    network_configuration = prepare_nework(args)
    # pretraining_configuration = prepare_pretraining(args)
    optimizaiton_configuration = prepare_optimizer(args)
    kfac_configuration = prepare_kfac(args)
    mcmc_configuration = prepare_mcmc(args)

    # some other keyword arguments?
    kwargs = {}

    NN.train(
        network_configuration,
        optimizaiton_configuration,
        kfac_configuration,
        mcmc_configuration,
        **kwargs
    )


if __name__ == '__main__':
    main()
