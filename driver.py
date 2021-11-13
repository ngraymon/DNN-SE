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
# from numpy.random import default_rng
# import tensorflow as tf


# local imports
import flags
from log_conf import log
# import kfac
# import hamiltonian as h_module
import fnn
from train import Train
import monte_carlo as MC
import quantum_mechanics as QM


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
    network_configuration = fnn.NetworkConfiguration(*args, **kwargs)

    return network_configuration


def prepare_optimizer(*args):
    """ x """
    kwargs = {}
    optimizaiton_configuration = fnn.OptimizerConfiguration(*args, **kwargs)

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

    """ create the Network object
    Currently the Train class assumes there is a network object and it needs to provide the following functions:
        - `network.parameters()`, which returns relevant network parameters for
        - `network.zero_grad()`
        - `network.forward(walkers)`, which assumes it takes a `walkers` object that is the return value from a `mcmc.create()` call
    """
    L, n_up = 5, 1
    e_pos = np.array([[1, 1, 1],  [-1, 1, 1]])
    n_pos = np.array([[0, 2, 1],  [0, 0, 1]])
    network = fnn.FermiNet(L, n_up, e_pos, n_pos, custom_h_sizes=False, num_determinants=2)
    # wavefn = model.forward()
    # print(wavefn)

    """ create the mcmc object
    Currently the Train class assumes there is a mcmc object and it needs to provide the following functions:
        - `mcmc.create()`, which returns the configurations for each electron (`walkers`)
    """
    mcmc = None

    """ create the Hamiltonian object
    Currently the Train class assumes there is a Hamiltonian object and it needs to provide the following functions:
        - `H.kinetic(phi, walkers)`, which returns the kinetic value as a torch tensor?
        - `H.potential(walkers)`, which returns the potential value as a torch tensor?
    """
    hamiltonian = None

    # create the Train object
    trainer_obj = Train(network, mcmc, hamiltonian, args)

    trainer_obj.train(
        network_configuration,
        optimizaiton_configuration,
        kfac_configuration,
        mcmc_configuration,
        **kwargs
    )


if __name__ == '__main__':
    main()
