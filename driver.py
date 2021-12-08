""" driver module

This module runs the program.

You can run it with
    python3 driver.py

"""

# system imports
import os
from os.path import abspath, join, dirname, isdir
import sys
import copy
from datetime import datetime as dt
import argparse
import time
from types import SimpleNamespace
from typing import Optional, Sequence

# third party imports
import numpy as np
# from numpy.random import default_rng
import torch


# local imports
from flags import flags
from log_conf import log, setLevelDebug, setLevelInfo
import kfac
import hamiltonian
import system
import fnn
import elements
from train import Train
from monte_carlo import MonteCarlo
# import plot_helium
# import quantum_mechanics as QM


tab = " "*4  # define tab as 4 spaces


# -------------------------- spin assignment functions ---------------------- #
def total_spin_assignment(spin_config_list):
    """ Return the total (up, down) spin assignment for a list. """

    # first we get the up/down spin assignments
    up_spin_list, down_spin_list = zip(*spin_config_list)

    # find the total number of electrons of each spin
    total_up, total_down = sum(up_spin_list), sum(down_spin_list)

    return (total_up, total_down)


def polarity_of_spin_list(spin_config_list):
    """ Return the total polarity of the list of spin configurations. """

    total_up, total_down = total_spin_assignment(spin_config_list)

    spin_config = (total_up, total_down)

    return spin_polarity(spin_config)


def correct_proposed_spins(proposed_spin_config_list, system_polarity_magnitude):
    """ Swap the values for each 2-length tuple in the given list.

    The objective is to swap the up/down spin assignments so that the
        polarities sign will be flipped.

    The function name is not the best, should be improved in the future.
    Good enough for now.
    """

    # first we get the up/down spin assignments
    up_spin_list, down_spin_list = zip(*proposed_spin_config_list)

    # find the total number of electrons of each spin
    total_up, total_down = sum(up_spin_list), sum(down_spin_list)

    # we need to swap all the up/down assignments (list changed through reference)
    proposed_spin_config_list = [(down, up) for up, down in zip(up_spin_list, down_spin_list)]

    # flip the sign on the polarity
    spin_config = (total_up, total_down)
    return -spin_polarity(spin_config)


def spin_polarity(electron_spin_assignment):
    """ Return the polarity.
    For a specific assignment of up & down spin electrons.
    Given (3 up spin, 4 down spin) the polarity is -1

    We simply subtract the number of down spin electrons from the number of up spin electrons
    """
    number_of_up_spin, number_of_down_spin = electron_spin_assignment
    return number_of_up_spin - number_of_down_spin


def greedy_guess_electron_spins(temp_atom_list, absolute_spin_polarity):
    """ Produce a list of tuples.
    For each atom in the `temp_atom_list` there is a corresponding tuple.
    Each tuple represents a spin assignment (up, down).
    For example [(1, 3), (2, 1), (2, 2), ] would be
        a possible return value given a `temp_atom_list` of length 3.
    """
    spin_configuration_list = []

    # keep track of the total polarity of our guess at assignment
    internal_spin_polarity = 0

    for ion in temp_atom_list:
        # Greedily assign up and down electrons based upon the ground state spin
        # configuration of an isolated atom.

        # we have not implemented the element module yet
        if False:
            unpaired_electrons = elements.ATOMIC_NUMS[ion.atomic_number].spin_config

        unpaired_electrons = 1

        """ To explain what is happening:
        Suppose we have an atom with 2 unpaired electrons and 10 total electrons

        Then
            n_A = (10 + 2) // 2 = 6
        and
            n_B = 10 - 6 = 4

        and so our spin configuration could be (6, 4) OR (4, 6)
            where (up, down) is what we are counting
        If the total polarity of the whole molecule is +/-1 then we will choose
            (4, 6)
            however if the polarity was +/-4 then the `new_spin`
            could be a negative number and we would end up with
            (6, 4) as the spin configuration
        """
        nof_electrons = ion.number
        n_A = (nof_electrons + unpaired_electrons) // 2
        n_B = nof_electrons - n_A

        # the new assignment of spin
        new_spin = internal_spin_polarity + unpaired_electrons - absolute_spin_polarity

        # Attempt to keep spin polarization as close to 0 as possible.
        if internal_spin_polarity > 0 and new_spin > 0:
            spin_config = [n_B, n_A]
        else:
            spin_config = [n_A, n_B]

        # record the contribution to the total spin polarity
        internal_spin_polarity += spin_polarity(spin_config)

        # record the spin choice
        spin_configuration_list.append(spin_config)

    return spin_configuration_list


def heuristically_set_initial_electron_positions(
    assigned_spin_polarity,
    absolute_spin_polarity,
    spin_configuration_list
    ):
    """ x """

    while assigned_spin_polarity != absolute_spin_polarity:

        atom_spin_pols = [abs(spin_polarity(spin)) for spin in spin_configuration_list]
        atom_index = atom_spin_pols.index(max(atom_spin_pols))
        n_up, n_down = spin_configuration_list[atom_index]

        # we exchange spin on individual electrons 1 at a time
        if n_up <= n_down:
            new_spin_assignment = (n_up + 1, n_down - 1)
        elif n_up < n_down:
            new_spin_assignment = (n_up - 1, n_down + 1)

        spin_configuration_list[atom_index] = new_spin_assignment

        # both polarity changes correspond to +/- 2 change
        # which agrees with the prior exchange of spin on 1 electron
        polarity_too_high = assigned_spin_polarity > absolute_spin_polarity
        polarity_too_low = assigned_spin_polarity < absolute_spin_polarity

        if polarity_too_high:
            assigned_spin_polarity += 2
        elif polarity_too_low:
            assigned_spin_polarity -= 2

    # we directly modified `spin_configuration_list` so no return value
    return


def generate_electron_position_vector(molecule, electron_spec):
    """Assigns electrons to atoms using non-interacting spin configurations.

    In the future I think its best to move this to the Hamiltonian module

    Args:
        molecule: List of Hamiltonian.Atom objects for each atom in the system.
        electron_spec: Pair of ints giving number of alpha (spin-up) and beta
            (spin-down) electrons.

    Returns:
        1D np.ndarray of length 3N containing initial mean positions of each
        electron based upon the atom positions, where N is the total number of
        electrons. The first 3*electrons[0] positions correspond to the alpha
        (spin-up) electrons and the next 3*electrons[1] to the beta (spin-down)
        electrons.

    Raises:
        RuntimeError: if a different number of electrons or different spin
        polarization is generated.

    'Note': spin polarization is a measurement of the alignment(agreement)
    of the electrons spin and the field/space being measured against.

    see the following for some more general information:
        - https://en.wikipedia.org/wiki/Spintronics
        - https://en.wikipedia.org/wiki/Spin_polarization
    """

    # just to make it absolutely clear what the `electron_spec` tuple represents
    nof_up_spin_electrons, nof_down_spin_electrons = electron_spec

    # Assign electrons based upon unperturbed atoms and ignore impact of
    # fractional nuclear charge.
    nuclei = [int(round(atom.charge)) for atom in molecule]

    # calculate the total charge
    total_charge = sum(nuclei) - (nof_up_spin_electrons + nof_down_spin_electrons)

    log.debug(f"Total charge of the molecule/system: {total_charge}")
    log.debug(f"{nof_up_spin_electrons = }")
    log.debug(f"{nof_down_spin_electrons = }")

    # Construct a dummy iso-electronic neutral system.
    temp_atom_list = [copy.copy(atom) for atom in molecule]

    if total_charge == 0:
        log.debug('System is already neutral')
    elif total_charge != 0:
        log.warning(
            'System has non-zero polarity.'
            'Using heuristics to set initial electron positions'
        )
        # calculate the individual atomic charge
        charge = 1 if total_charge > 0 else -1

    """ we want to have a neutral total charge
    so we go through the list of nuclei/atoms in order of highest to lowest charge
    and subtract (-1/+1) from that nuclei/atoms AS WELL as the total charge
        repeat until the total charge is 0
    """
    while total_charge != 0:

        # take the next atom with the lowest (positive/negative) charge
        lowest_absolute_charge = max(nuclei) if total_charge < 0 else min(nuclei)

        # get its associated index
        lowest_atom_index = nuclei.index(lowest_absolute_charge)

        # retrieve the object
        largest_atom = temp_atom_list[lowest_atom_index]

        # calculate a new charge
        new_charge = largest_atom.charge - charge

        # get an integer value for the atomic number
        new_atomic_number = int(round(largest_atom.charge))

        # if the charge of that atom is now zero then remove it from our
        # "dummy" neutral system
        if int(round(new_charge)) == 0:
            temp_atom_list.pop(lowest_atom_index)

        # otherwise we replace that atom with a new atom at the modified charge
        else:
            # assign a symbol to this `atom` from the element table
            # that is consistent with the new element that it represents
            log.debug(f"{new_atomic_number = }")
            if False:
                new_symbol = elements.ATOMIC_NUMS[new_atomic_number].symbol

            if new_atomic_number == 1:
                new_symbol = 'H'
            elif new_atomic_number == 6:
                new_symbol = 'C'
            else:
                raise Exception("cannot support anything else a.t.m.")

            """the new atom
            since I'm currently using a namedtuple
            we must create a new one each time
            probably best to move to SimpleNamespace later
            once we want to move to helium
            """
            new_atom = system.Atom(
                new_symbol,
                new_atomic_number,
                largest_atom.coords,
                new_charge
            )

            # replace the old atom in the `temp_atom_list`
            molecule[lowest_atom_index] = new_atom

        # update the total charge
        total_charge -= charge

        # make sure to update the charge for the appropriate atom in the
        # nuclei list
        nuclei[lowest_atom_index] = new_charge

    absolute_spin_polarity = abs(spin_polarity(electron_spec))

    if temp_atom_list == []:
        raise Exception(f"The molecule is malformed please check your input:\n{molecule}")

    # if there is just 1 atom left then we don't have to fiddle with the number
    # of electron spins (up/down)
    elif len(temp_atom_list) == 1:
        spin_config_list = [electron_spec]

    # otherwise we need to re-assign the up/down spin values of the electrons
    elif len(temp_atom_list) > 1:
        spin_config_list = greedy_guess_electron_spins(temp_atom_list, absolute_spin_polarity)

    # by this point we hope to have correctly assigned up/down spins

    # First we check if the sign is negative/zero (I think Jame's code has a bug here??? )
    assigned_spin_polarity = polarity_of_spin_list(spin_config_list)
    if int(np.sign(assigned_spin_polarity)) in [0, -1]:
        assigned_spin_polarity = correct_proposed_spins(spin_config_list, electron_spec)

    # If the polarity is still not the correct magnitude then we need to use heuristics
    if assigned_spin_polarity != absolute_spin_polarity:
        log.debug(
            'Spin polarization does not match isolated atoms. '
            'Using heuristics to set initial electron positions.'
        )

        # attempt to correct spins with heuristics
        spin_config_list = heuristically_set_initial_electron_positions(
            assigned_spin_polarity, absolute_spin_polarity, spin_config_list
        )

    # Check the spin polarity AGAIN and correct if needed
    if polarity_of_spin_list(spin_config_list) == -spin_polarity(electron_spec):
        assigned_spin_polarity = correct_proposed_spins(spin_config_list, electron_spec)

    # log the final assignment
    iterator = zip(molecule, spin_config_list)
    lst = [f"{atom.symbol}: {spin_config}" for atom, spin_config in iterator]
    log.debug(f"\nElectrons assigned:\n{tab}{', '.join(lst)}.")

    # NOTE for future me: original code seemed to be designed for handling lists of lists
    proposed_electron_spec = total_spin_assignment(spin_config_list)

    nof_up, nof_down = electron_spec
    new_nof_up, new_nof_down = proposed_electron_spec

    # confirm that the proposed number of up-spin electrons and down-spin electrons
    # agrees with the original numbers, respectively
    if new_nof_up != nof_up or new_nof_down != nof_down:
        raise RuntimeError(
            "Assigned incorrect number of electrons "
            f" ({new_nof_up}, {new_nof_down}) instead of ({nof_up}, {nof_down})"
        )

    # all spin assignments need to be non-negative integers
    if any(0 > min(spin_config) for spin_config in zip(*spin_config_list)):
        raise RuntimeError('Assigned negative number of electrons!')

    up_spin_list = [np.tile(atom.coords, e[0]) for atom, e in zip(temp_atom_list, spin_config_list)]
    down_spin_list = [np.tile(atom.coords, e[1]) for atom, e in zip(temp_atom_list, spin_config_list)]

    up_spin_string = f"\n{tab}".join([repr(spin) for spin in up_spin_list])
    down_spin_string = f"\n{tab}".join([repr(spin) for spin in down_spin_list])
    log.debug(f"\nup_spin_list:\n{tab}{up_spin_string}")
    log.debug(f"\ndown_spin_list:\n{tab}{down_spin_string}")

    both_spins_list = np.concatenate(up_spin_list + down_spin_list)

    walker_string = f"\n{tab}".join([
        f"electron {i+1} : {both_spins_list[i*3:(i+1)*3]}"
        for i in range(len(both_spins_list) // 3)
    ])

    log.debug(f"\nwalker tensor:\n{tab}{walker_string}")

    # glue the two lists together
    electron_positions = np.array(both_spins_list).reshape((sum(electron_spec), 3))

    return electron_positions

# --------------------------- configuration functions ----------------------- #


def prepare_file_paths():
    """ x """

    try:
        if not isdir(flags.result_folder):
            os.makedirs(flags.result_folder, exist_ok=True)
    except Exception as e:
        print(
            f"Failed to create results folder at {flags.result_folder}\n"
            "Is there an issue with the OS and/or file system?"
        )
        raise e

    time = dt.now().strftime("%b_%d_%Hh_%Mm_%Ss")

    result_path = join(flags.result_folder, f'results_{time}')

    try:
        if not isdir(result_path):
            os.makedirs(result_path, exist_ok=True)
    except Exception as e:
        print(
            f"Failed to create results folder at {flags.result_folder}\n"
            "Is there an issue with the OS and/or file system?"
        )
        raise e

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


def prepare_nework_configuration(*args):
    """Network configuration for Fermi Net.

    Attributes:
        determinants: Number of determinants to use.
        use_envelope: Include multiplicative exponentially-decaying envelopes on
            each orbital. Calculations will not converge if set to False.
    """
    config = SimpleNamespace()  # temporary solution

    config.determinants: int = 16
    config.use_envelope: bool = False

    return config


def prepare_pretraining_configuration(*args):
    """Hartree-Fock pretraining algorithm configuration for Fermi Nets.

    Attributes:
        iterations: Number of iterations for which to pretrain the network to match
            Hartree-Fock orbitals.
        basis: Basis set used to run Hartree-Fock calculation in PySCF.
    """
    config = SimpleNamespace()  # temporary solution

    config.iterations: int = 1000
    config.basis: str = 'sto-3g'

    return config


def prepare_optimizer_configuration(*args):
    """ Need to discuss as group how we are implementing the
    'configs/preparation'
    """
    kwargs = {}
    try:
        optimization_configuration = fnn.OptimizerConfiguration(*args, **kwargs)
    except AttributeError as e:
        log.debug('Optimization configuration not implemented')
        optimization_configuration = None

    return optimization_configuration


def prepare_kfac_configuration(*args):
    """ Need to discuss as group how we are implementing the
    'configs/preparation'
    """
    kwargs = {}
    try:
        kfac_configuration = kfac.KfacConfiguration(*args, **kwargs)
    except AttributeError as e:
        log.debug('KFAC configuration not implemented')
        kfac_configuration = None

    return kfac_configuration


def prepare_mcmc_configuration(*args):
    """Markov Chain Monte Carlo configuration for Fermi Nets.

    Attributes:
        burn_in: Number of burn in steps after pretraining.
        steps: 'Number of MCMC steps to make between network updates.
        init_stddev: Width of (atom-centred) Gaussians used to generate initial
            electron configurations.
        move_stddev: Width of Gaussian used for random moves.
        init_offset: Iterable of 3*nof_electrons giving the mean initial position of
            each electron (the offset from 0). Configurations are drawn using Gaussians
            of stddev `init_stddev` at each 3D position. Alpha electrons are listed before beta
            electrons. If empty, electrons are assigned to atoms based upon the
            isolated atom spin configuration. Expert use only.
    """
    config = SimpleNamespace()   # temporary solution

    config.burn_in = 100
    config.steps = 10
    config.init_stddev = 0.8
    config.move_stddev = 0.02
    config.init_offset: Optional[Sequence[float]] = None

    return config

# -------------------------- initialization functions ----------------------- #


def prepare_trainer(network, mcmc, hamiltonian_operators, hartree_fock=None):
    """ create the Train object """
    # param = {'lr': 0.01, 'epoch': 100}
    param = {'lr': flags.learning_rate, 'epoch': flags.iterations}

    # the current implementation
    if hartree_fock is None:
        return Train(network, mcmc, hamiltonian_operators, param)

    # possible future approach
    return Train(network, mcmc, hamiltonian_operators, hartree_fock, param)


def prepare_scf(molecule, spin_config, config, using_scf_flag=False):
    """ create the SCF object
    This currently doesn't do anything as we don't have a SCF module
    implemented yet.
    """
    if not using_scf_flag:
        return None

    scf_kwargs = {
        'nof_electrons': sum(spin_config),
        'restricted': False,
        'basis': config.basis
    }

    # here is where we would initialize the scf object
    scf_object = tuple(molecule, scf_kwargs)

    # preform pretraining if requested
    if config.iterations > 0:
        scf_object.run()

    return scf_object


def prepare_network(molecule, nof_electrons, spin_config):
    """ create the Network object
    Currently the Train class assumes there is a network object and it needs to provide the following functions:
        - `network.parameters()`, which returns relevant network parameters for
        - `network.zero_grad()`
        - `network.forward(walkers)`, which assumes it takes a `walkers` object that is the return value from a `mcmc.create()` call
    """
    number_of_up_spin_electrons, number_of_down_spin_electrons = spin_config

    # create nuclear positions
    nuclear_positions = torch.zeros((len(molecule), 3))

    # temporarily create numpy array
    temporary_arr = np.array([np.array(atom.coords) for atom in molecule])

    # wrap that in a pytorch tensor (necessary for backpropagation of the network)
    nuclear_positions[:] = torch.tensor(temporary_arr)

    # dummy electron positions for now
    electron_positions = torch.zeros((sum(spin_config), 3))

    return fnn.FermiNet(
        spin_config,
        electron_positions,
        nuclear_positions,
        flags.hidden_units,
        num_determinants=flags.determinants
    )


def prepare_hamiltonian(molecule, nof_electrons):
    """ create the Hamiltonian object
    Currently the Train class assumes there is a Hamiltonian object and it needs to provide the following functions:
        - `H.kinetic(phi, walkers)`, which returns the kinetic value as a torch tensor?
        - `H.potential(walkers)`, which returns the potential value as a torch tensor?
    """
    return hamiltonian.operators(molecule, nof_electrons)


def prepare_mean_array(config, molecule, spin_config):
    """ Check that the electron positions are valid (and fixes them if not) """

    # first we need to generate the offsets/means which define the distribution
    # from which we sample our initial walker configurations
    init_offset = config.init_offset

    # if no offsets were provided then we need to calculate them
    if init_offset is None:
        log.info("Attempting to generate new electron positions")
        init_offset = generate_electron_position_vector(molecule, spin_config)
        return init_offset

    # if offsets were provided
    else:
        nof_electrons = sum(spin_config)
        # check that we have 3*N init_offsets
        if len(init_offset) == 3 * nof_electrons:
            """
            for some reason we cast each mean to a float?
            are we upcasting from 32 -> 64 here?
            or were they ints?
            unclear...
            """
            init_offset = [float(x) for x in init_offset]

        # otherwise we cannot proceed
        else:
            raise RuntimeError(
                "Initial electron positions of incorrect shape. "
                f"({init_offset} not {3 * nof_electrons})"
            )


def prepare_mcmc(config, network_object, init_offset, precision):
    """ Return an mcmc object for training the network.

    Currently the Train class assumes there is a mcmc object and it needs to provide the following functions:
        - `mcmc.create()`, which returns the configurations for each electron (`walkers`)

    This function initializes the mcmc object with appropriate flags/parameters
    """
    mcmc_object = MonteCarlo(
        network=network_object,
        batch_size=flags.batch_size,
        initial_offset=init_offset,
        initial_stddev=config.init_stddev,
        offset=flags.mcmc_offset,  # this one needs more inspection
        stddev=config.move_stddev,
        nof_steps=flags.mcmc_steps_per_update,
        dtype=precision,
    )
    return mcmc_object

# ------------------------------ main function ------------------------------ #


def main(molecule, spin_config):
    """ Wrapper function for Train.train().

    Handles all the finicky details of setting up objects/classes.
    Initializes appropriate parameters.
    Handles different cases (multi-gpu/single-gpu/cpu)
    """

    print(f"Running on device: {flags.device}")

    if flags.deterministic:
        # set random seed to flags.deterministic_seed
        log.info('Running in deterministic mode. Performance will be reduced.')

    """ !NOTE! - so far these don't really do anything
    they are more necessary in the event when we need to do multi-gpu stuff
    and/or if we are having more configurational parameters
    """

    result_path = prepare_file_paths()
    args = prepare_system(result_path)
    network_config = prepare_nework_configuration(args)
    pretraining_config = prepare_pretraining_configuration(args)
    optimizaiton_config = prepare_optimizer_configuration(args)
    kfac_config = prepare_kfac_configuration(args)
    mcmc_config = prepare_mcmc_configuration()
    log.debug("Finished preparing the config parameters")

    # some other keyword arguments?
    kwargs = {}
    nof_electrons = sum(spin_config)
    # probably want to use pytorch floats?
    precision = torch.float64 if flags.double_precision else torch.float32

    # temporary work around until we have scf implemented
    using_scf_flag = False

    # create the network object
    network_object = prepare_network(molecule, nof_electrons, spin_config).to(flags.device)
    log.debug("Finished initializing the Network object")

    # currently only returns `None`
    scf_object = prepare_scf(molecule, spin_config, pretraining_config, using_scf_flag=False)
    log.debug("Finished initializing the SCF object")

    # create the Hamiltonian operators
    hamiltonian_operators = prepare_hamiltonian(molecule, nof_electrons)
    log.debug("Finished initializing the Hamiltonian operators")

    # create initial means/offsets to define the sampling's normal distribution
    init_means = prepare_mean_array(mcmc_config, molecule, spin_config)

    # create the Monte Carlo object
    mcmc_object = prepare_mcmc(mcmc_config, network_object, init_means, precision)
    log.debug("Finished initializing the MCMC object")

    # we might do this in the future
    # (i believe this part of the code was to generate HF data to compare to)
    # (possibly for burn in...? still unclear)
    if using_scf_flag:
        hf_object = prepare_mcmc(mcmc_config, scf_object, init_means, precision)
    else:
        hf_object = None
    log.debug("Finished initializing the HF object")

    # create the trainer object
    trainer_object = prepare_trainer(
        network_object, mcmc_object, hamiltonian_operators, hf_object
    )
    log.debug("Finished initializing the trainer object")

    # what we've all been waiting for, the ACTUAL training!
    log.debug("Attempting to start the training")
    train_start_time = time.time()
    loss_storage = trainer_object.train(
        # network_configuration,
        # optimizaiton_configuration,
        # kfac_configuration,
        # mcmc_configuration,
        **kwargs
    )
    train_stop_time = time.time()
    log.info("Training completed\t [{0:.4f} s]".format(train_stop_time - train_start_time))
    # Save:
    network_object.save(result_path)
    log.info("Model Saved.")
    with open(join(result_path, "time.txt"), 'w') as f:
        f.write(str(train_stop_time - train_start_time))
        f.close()
    with open(join(result_path, "loss_storage.txt"), 'w') as f:
        f.write(str(loss_storage))
        f.close()

    # plot the training progress
    # plotting.plot_training_metrics(loss_storage, name)

    print("Success!")

    return loss_storage


def simple_plotting(loss_storage):
    """ x """
    import matplotlib.pyplot as plt
    plt.plot(loss_storage)
    plt.show()
    # plotting.plot_helium(network_object, name)


def prepare_parsed_arguments():
    """ Wrapper for argparser setup """

    # setLevelDebug()

    # use input to specify system for debugging purposes
    # name = str(sys.argv[1]) if len(sys.argv) > 1 else 'hydrogen'
    # number = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # formatclass = argparse.RawDescriptionHelpFormatter
    # formatclass = argparse.RawTextHelpFormatter
    formatclass = argparse.ArgumentDefaultsHelpFormatter  # I liked this the best
    # formatclass = argparse.MetavarTypeHelpFormatter

    # parse the arguments
    parser = argparse.ArgumentParser(description="Fermi net 2.0?", formatter_class=formatclass)
    parser.add_argument('--name', type=str, default='hydrogen', metavar='system name', help='the name of the QM system to evaluate')
    parser.add_argument('--length', type=int, default=1, metavar='length of chain', help='if using a hydrogen chain, how long the chain is')
    parser.add_argument('--param', type=str, default='param.json', metavar='param.json', help='file name for json attributes')
    parser.add_argument('-res-path', type=str, default=flags.result_folder, metavar='results_dir', help='path to save the plots at')
    parser.add_argument('-n', type=int, default=flags.batch_size, metavar='number_of_replicas', help='number of replica state vectors for mc to propagate')
    parser.add_argument('-v', type=int, default=1, metavar='N', help='verbosity (set to 2 for full debugging)')
    parser.add_argument('-device', type=str, default=flags.device, metavar='device_name', help='specify a specific device for PyTorch to use')
    parser.add_argument('-epochs', type=int, default=flags.iterations, metavar='num_epochs', help='number of epochs to run for')
    parser.add_argument('-lr', type=float, default=flags.learning_rate, metavar='learning_rate', help='learning rate for the optimiser')

    return parser.parse_args()


def prepare_molecule_and_spins(pargs):
    """ x """

    # hydrogen (simplest test case)
    if pargs.name == "hydrogen":
        molecule, spins = system.Atom.build_basic_atom(symbol='H', charge=0)

    # for testing a simple system with _at least_ 2 atoms
    elif pargs.name == "chain":
        molecule, spins = system.hydrogen_chains(n=pargs.length, width=0.5)

    # for testing dimensionality of the walker tensor
    elif pargs.name == "methane":
        molecule, spins = system.methane()

    elif pargs.name == "helium":
        raise Exception('Not supported yet')
        molecule, spins = system.Atom.build_basic_atom(symbol='He', charge=0)

    else:
        raise Exception(f"{pargs.name} is not a supported system yet")

    print("The system input is as follows:")
    for i, m in enumerate(molecule):
        print(f"  Atom {i}: {m}")
    print(f"  {spins = }\n")

    return molecule, spins


if __name__ == '__main__':

    # process the users input
    pargs = prepare_parsed_arguments()

    #the defaults in the argparse are the flags values, so if the args are omitted it follows flags.py
    flags.device = pargs.device
    flags.iterations = pargs.epochs
    flags.learning_rate = pargs.lr

    if pargs.v == 2:
        setLevelDebug()

    # prepare the system
    molecule, spins = prepare_molecule_and_spins(pargs)

    loss_storage = main(molecule, spins)

    if False:  # make this an argparse parameter
        simple_plotting(loss_storage)
