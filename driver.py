""" driver module

This module runs the program.

You can run it with
    python3 driver.py

"""

# system imports
import copy
import os
from os.path import abspath, join, dirname, isdir
import sys
from datetime import datetime as dt
from types import SimpleNamespace
from typing import Optional, Sequence

# third party imports
import numpy as np
# from numpy.random import default_rng
import torch


# local imports
from flags import flags
from log_conf import log
import kfac
import hamiltonian
import fnn
import elements
from train import Train
from monte_carlo import MonteCarlo
import quantum_mechanics as QM


def generate_electron_position_vector(molecule, electrons):
    """Assigns electrons to atoms using non-interacting spin configurations.

    In the future I think its best to move this to the Hamiltonian module

    Args:
        molecule: List of Hamiltonian.Atom objects for each atom in the system.
        electrons: Pair of ints giving number of alpha (spin-up) and beta
            (spin-down) electrons.

    Returns:
        1D np.ndarray of length 3N containing initial mean positions of each
        electron based upon the atom positions, where N is the total number of
        electrons. The first 3*electrons[0] positions correspond to the alpha
        (spin-up) electrons and the next 3*electrons[1] to the beta (spin-down)
        electrons.

    Raises:
        RuntimeError: if a different number of electrons or different spin
        polarisation is generated.
    """

    # Assign electrons based upon unperturbed atoms and ignore impact of
    # fractional nuclear charge.
    nuclei = [int(round(atom.charge)) for atom in molecule]

    # calculate the total charge
    total_charge = sum(nuclei) - sum(electrons)

    # Construct a dummy iso-electronic neutral system.
    neutral_molecule = [copy.copy(atom) for atom in molecule]

    if total_charge != 0:
        log.warning(
            'Charged system.'
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

        # take the next atom with the highest (positive/negative) charge
        atom_index = nuclei.index(max(nuclei) if total_charge < 0 else min(nuclei))
        atom = neutral_molecule[atom_index]
        atom.charge -= charge
        atom.atomic_number = int(round(atom.charge))

        # if the charge of that atom is now zero then remove it from our
        # "dummy" neutral system
        if int(round(atom.charge)) == 0:
            neutral_molecule.pop(atom_index)
        # otherwise assign a symbol to this `atom` from the element table
        else:
            atom.symbol = elements.ATOMIC_NUMS[atom.atomic_number].symbol

        # update the total charge
        total_charge -= charge

        # rebuild the list with/without that atom we may have removed
        nuclei = [int(round(atom.charge)) for atom in neutral_molecule]

    #
    spin_pol = lambda electrons: electrons[0] - electrons[1]
    abs_spin_pol = abs(spin_pol(electrons))

    if len(neutral_molecule) == 1:
        elecs_atom = [electrons]
    else:
        elecs_atom = []
        spin_pol_assigned = 0

        for ion in neutral_molecule:
            # Greedily assign up and down electrons based upon the ground state spin
            # configuration of an isolated atom.
            atom_spin_pol = elements.ATOMIC_NUMS[ion.atomic_number].spin_config
            nelec = ion.atomic_number
            na = (nelec + atom_spin_pol) // 2
            nb = nelec - na

            # Attempt to keep spin polarisation as close to 0 as possible.
            if (
                spin_pol_assigned > 0
                and spin_pol_assigned + atom_spin_pol > abs_spin_pol
            ):
                elec_atom = [nb, na]
            else:
                elec_atom = [na, nb]

            spin_pol_assigned += spin_pol(elec_atom)
            elecs_atom.append(elec_atom)

    electrons_assigned = [sum(e) for e in zip(*elecs_atom)]
    spin_pol_assigned = spin_pol(electrons_assigned)

    if np.sign(spin_pol_assigned) == -np.sign(abs_spin_pol):
        # Started with the wrong guess for spin-up vs spin-down.
        elecs_atom = [e[::-1] for e in elecs_atom]
        spin_pol_assigned = -spin_pol_assigned

    if spin_pol_assigned != abs_spin_pol:
        log.info(
            'Spin polarisation does not match isolated atoms. '
            'Using heuristics to set initial electron positions.'
        )

    while spin_pol_assigned != abs_spin_pol:

        atom_spin_pols = [abs(spin_pol(e)) for e in elecs_atom]
        atom_index = atom_spin_pols.index(max(atom_spin_pols))
        elec_atom = elecs_atom[atom_index]

        if spin_pol_assigned < abs_spin_pol and elec_atom[0] <= elec_atom[1]:
            elec_atom[0] += 1
            elec_atom[1] -= 1
            spin_pol_assigned += 2

        elif spin_pol_assigned < abs_spin_pol and elec_atom[0] > elec_atom[1]:
            elec_atom[0] -= 1
            elec_atom[1] += 1
            spin_pol_assigned += 2

        elif spin_pol_assigned > abs_spin_pol and elec_atom[0] > elec_atom[1]:
            elec_atom[0] -= 1
            elec_atom[1] += 1
            spin_pol_assigned -= 2

        else:
            elec_atom[0] += 1
            elec_atom[1] -= 1
            spin_pol_assigned -= 2

    electrons_assigned = [sum(e) for e in zip(*elecs_atom)]

    if spin_pol(electrons_assigned) == -spin_pol(electrons):
        elecs_atom = [e[::-1] for e in elecs_atom]
        electrons_assigned = electrons_assigned[::-1]

    log_string = ', '.join(
        [f"{atom.symbol}: {elec_atom}" for atom, elec_atom in zip(molecule, elecs_atom)]
    )
    log.info(f"Electrons assigned {log_string}.")

    if any(e != e_assign for e, e_assign in zip(electrons, electrons_assigned)):
        raise RuntimeError(
            "Assigned incorrect number of electrons"
            f"([{electrons_assigned} instead of {electrons}]"
        )

    if any(min(ne) < 0 for ne in zip(*elecs_atom)):
        raise RuntimeError('Assigned negative number of electrons!')

    electron_positions = np.concatenate(
        [
          np.tile(atom.coords, e[0]) for atom, e in zip(neutral_molecule, elecs_atom)
        ] + [
          np.tile(atom.coords, e[1]) for atom, e in zip(neutral_molecule, elecs_atom)
        ]
    )

    return electron_positions


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

    os.mkdir(result_path)

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

# --------------------------- configuration functions ----------------------- #


def prepare_nework_configuration(*args):
    """Network configuration for Fermi Net.

    Attributes:
        determinants: Number of determinants to use.
        use_envelope: Include multiplicative exponentially-decaying envelopes on
            each orbital. Calculations will not converge if set to False.
    """
    config = SimpleNamespace()  # temporary solution

    # config.hidden_units/layers = ()
    config.determinants: int = 16
    config.use_envelope: bool = False

    return config


def pretraining_config(*args):
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
        print('Optimization configuration not implemented')
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
        print('KFAC configuration not implemented')
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
        init_offset: Iterable of 3*nelectrons giving the mean initial position of
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
    param = {'lr': 0.01, 'epoch': 10}

    # the current implementation
    if hartree_fock is None:
        return Train(network, mcmc, hamiltonian_operators, param)

    # possible future approach
    return Train(network, mcmc, hamiltonian_operators, hartree_fock, param)


def prepare_scf(molecule, spins, config, using_scf_flag=False):
    """ create the SCF object
    This currently doesn't do anything as we don't have a SCF module
    implemented yet.
    """
    if not using_scf_flag:
        return None

    scf_kwargs = {
        'nelectrons': spins,
        'restricted': False,
        'basis': config.basis
    }

    # here is where we would initialize the scf object
    scf_object = tuple(molecule, scf_kwargs)

    # preform pretraining if requested
    if config.iterations > 0:
        scf_object.run()

    return scf_object


def prepare_network(molecule, nelectrons):
    """ create the Network object
    Currently the Train class assumes there is a network object and it needs to provide the following functions:
        - `network.parameters()`, which returns relevant network parameters for
        - `network.zero_grad()`
        - `network.forward(walkers)`, which assumes it takes a `walkers` object that is the return value from a `mcmc.create()` call
    """
    L, n_up = 5, 1
    e_pos = np.array([[1, 1, 1],  [-1, 1, 1]])
    n_pos = np.array([[0, 2, 1],  [0, 0, 1]])

    return fnn.FermiNet(L, n_up, e_pos, n_pos, custom_h_sizes=False, num_determinants=2)


def prepare_hamiltonian(molecule, nelectrons):
    """ create the Hamiltonian object
    Currently the Train class assumes there is a Hamiltonian object and it needs to provide the following functions:
        - `H.kinetic(phi, walkers)`, which returns the kinetic value as a torch tensor?
        - `H.potential(walkers)`, which returns the potential value as a torch tensor?
    """
    return hamiltonian.operators(molecule, nelectrons)


def prepare_mcmc(config, network_object, nelectrons, precision):
    """ Return an mcmc object for training the network.

    Currently the Train class assumes there is a mcmc object and it needs to provide the following functions:
        - `mcmc.create()`, which returns the configurations for each electron (`walkers`)

    This function does two things:
      - check that the electron positions are valid (and fixes them if not)
      - initializes the mcmc object with appropriate flags/parameters
    """

    # first we need to generate the offsets/means which define the distribution
    # from which we sample our initial walker configurations
    init_offset = config.init_offset

    # if no offsets were provided then we need to calculate them
    if init_offset is None:
        init_offset = generate_electron_position_vector(molecule, spins)

    # if offsets were provided
    else:
        # check that we have 3N init_offsets
        if len(init_offset) == 3 * nelectrons:
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
                f"({init_offset} not {3 * nelectrons})"
            )

    # then we initialize the Monte Carlo object
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


def main(molecule, spins):
    """ Wrapper function for Train.train().

    Handles all the finicky details of setting up objects/classes.
    Initializes appropriate parameters.
    Handles different cases (multi-gpu/single-gpu/cpu)
    """

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
    pretraining_config = prepare_pretraining(args)
    optimizaiton_config = prepare_optimizer_configuration(args)
    kfac_config = prepare_kfac_configuration(args)
    mcmc_config = prepare_mcmc_configuration()

    # some other keyword arguments?
    kwargs = {}
    nelectrons = sum(spins)
    # probably want to use pytorch floats?
    precision = torch.float64 if flags.double_precision else torch.float32

    # temporary work around until we have scf implemented
    using_scf_flag = False

    # create the network object
    network_object = prepare_network()

    # currently only returns `None`
    scf_object = prepare_scf(molecule, spins, pretraining_config, using_scf_flag=False)

    # create the Hamiltonian operators
    hamiltonian_operators = prepare_hamiltonian(molecule, nelectrons)

    # create the Monte Carlo object
    mcmc_object = prepare_mcmc(mcmc_config, network_object, nelectrons, precision)

    # we might do this in the future
    # (i believe this part of the code was to generate HF data to compare to)
    # (possibly for burn in...? still unclear)
    if using_scf_flag:
        hf_object = prepare_mcmc(mcmc_config, scf_object, nelectrons, precision)
    else:
        hf_object = None

    # create the trainer object
    trainer_object = prepare_trainer(
        network_object, mcmc_object, hamiltonian_operators, hf_object
    )

    # what we've all been waiting for, the ACTUAL training!
    trainer_object.train(
        # network_configuration,
        # optimizaiton_configuration,
        # kfac_configuration,
        # mcmc_configuration,
        **kwargs
    )

    print("Success!")


if __name__ == '__main__':
    # what do the molecules look like?
    molecule = None
    # what do the spins look like?
    spins = None

    main(molecule, spins)
