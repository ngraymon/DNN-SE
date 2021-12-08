""" original_flags module
This module contains the definition of a SimpleNamespace object `original_flags` and initializes it with all the default values.
We're starting with a SimpleNamespace for simplicity.
https://docs.python.org/3/library/types.html#types.SimpleNamespace
We could also use namedtuples's:
https://docs.python.org/3/library/collections.html#collections.namedtuple

In FNN they use googles abseil, but starting with that package will probably present a roadblock to getting a MVP
so it is advisable to hold off for now.
https://abseil.io/
"""

# system imports
from os.path import abspath, dirname, join
from types import SimpleNamespace
import numpy as np

# third party imports

# local imports


""" simple instantiation
Use `original_flags` for testing and implementation
Make `original_original_flags` a direct copy of the FNN default flag settings for use in replicating their results.
"""
flags = SimpleNamespace()
original_flags = SimpleNamespace()


# ------------
# These are the flags we will be using for our simulation
"""these are values that may or may not be modified and then implented into the network
if any values are missing feel free to add to releavent section or let me (Matt)
know"""

flags.multi_gpu = False  # number of GPU's used
flags.batch_size = 7    # number of walkers
flags.double_precision = False    # the double or single precision
flags.pretrain_iterations = 1000  # the iterations over which the network is pretrained
flags.pretrain_basis = 'sto-3g'   # To do with PySCF and using the Hartree-Fock calculation

# optimisation flags
flags.iterations = 1000  # number of iterations
flags.clip_el = 5.0  # the scale at with to clip local energy????
flags.learning_rate = 1e-3  # learning rate
flags.learning_rate_decay = 1.0  # exponent of learning rate decay
                                 # can be changed to include exponential i.e. 1E1
flags.learning_rate_delay = 10000.0  # the scale of the rate of decay

# Flags for KFAC
"""KFAC isn't implemented at the moment but if it is
these are the parameters, ignore if not using KFAC"""
"""Need to impliment inital flags for ADAM is using that, must talk to JB"""

flags.use_kfac = False
flags.kfac_invert_every = 1
flags.kfac_cov_update_every = 1
flags.kfac_damping = 0.001
flags.kfac_coov_ema_decay = 0.95
flags.kfac_momentum_type = 'regular'
flags.kfac_adapt_damping = False
flags.kfac_damping_adaption_decay = 0.9
flags.kfac_damping_adaptation_interval = 5
flags.kfac_min_damping = 1e-4
flags._kfac_norm_contraint = 0.001

# flags for system

flags.system_charge = 0  # we will only be working with atoms so this will always be 0
flags.system_dim = 3  # dimensions of the system
# hydrogen circle, are we going to be implementing any hydrogen circles???
flags.system_type = 'molecule'  # shouldn't this be an input value? ... unclear (may not need this flag)
flags.system = "H"  # We will only be working with atoms so use the atomic symbol
                    # if we use molecules then the system should be the molecular symbol
                    # if "hn" the number of atoms in the hydrogen chain
flags.units = "bohr"  # the input units of the system. Check with everyone to see what our system uses

"""are we going to be using diatomics, hydrogen chains and hydrogen circles
if we are then these are the flags relating to these systems, if not simply ignore"""


flags.system_separation = 0.0  # for the hydrogen chain and diatomics, this is the separation between nuclei
                               # for Hydrogen circles, this is the radius of the circle
                               # for diatomics this will default to the bond length when 0

flags.system_angle = np.pi/4.0  # Angle from the x-axis for the hydrogen circle


#  flags for logging
"""This is to do with saving and logging results. More flags maybe
rrequired depending on how this is implemented"""
flags.deterministic = False
flags.result_folder = abspath(join(dirname(__file__), 'results')) # where and what the results folder is located and named
flags.stats_frequency = 1  # iterations between stat logging
flags.save_frequency = 10.0  # time (minutes??) between saving network parameters
flags.log_walkers = False  # whether to log walker values after every iteration. Comes with high data usage warning
flags.log_local_energies = False  # whether to log all local energies for each walker at each step
flags.log_wavefunction = False  # the same as above but for the wavefunction instead
flags.random_seed = 1


#  the name of the file where we save command line arguments
flags.cmdline_file = 'command_line_arguments.txt'


# flags for MCMC chain
""" number of burn in steps after pretraining
if 0 will not burn in or reinitialize walkers """
flags.mcmc_burn_in = 100
flags.mcmc_steps_per_update = 10  # number of mcmc steps between each update
flags.mcmc_offset = 0.0  # isn't actually used
flags.mcmc_initial_stddev = 0.8  # Gaussian used to generate the initial electron configuration
flags.mcmc_move_width = 0.2  # Width of the Gaussian used for the random moves

""" If left empty, the electrons are assigned to atoms based on isolated spin configuration
    The initial configurations of the the 3*nelectrons and their positions """
flags.mcmc_init_means = ''

# flags for Will's Pseudocode
"""Must ask Will how many of these are actually required
 and what values would be suitable
there may be crossover with some of the flags already
 present so only some alterations would be required"""
flags.fnn_electron_positions = None
flags.fnn_nuclei_positions = None
flags.fnn_n_up = None
flags.fnn_L = None
flags.fermilayer_h_in_dims = None
flags.fermilayer_h_out_dims = None


# flags for Andrew's Pseudocode
"""Must ask Andrew how many of these are actually required
 and what values would be suitable
there may be crossover with some of the flags already
 present so only some alterations would be required"""
flags.atoms = None
flags.nelectrons = None
flags.potential_epsilon = None
flags.x = None
flags.xs = None


# flags for the network architecture

# we will be running ferminet exclusively I believe but here is where different networks would be chosen
flags.network_architecture = 'ferminet'

""" The number of hidden units in each layer of the network
For Ferminet, the first number in each tuple is the number
units in the 1-electron stream and the second number is
the units in the 2 electron stream
"""
flags.hidden_units = [[32, 4],  [32, 4],  [32, 4],  [32, 4]]

flags.determinants = 16  # number of determinents in the ferminet
flags.r12_distance_between_particle = True  # include r12/distance features between electrons and nuclei
flags.r12_distance_between_electrons = True  # same as above but between electron pairs
flags.pos_ee_features = True  # Include electron-electron position features
flags.use_envelope = True  # include multiplicative exponentially decaying envelope
flags.residual = True  # Use residual connections
flags.after_det = 1  # Coma-separated configuration of neural network after the determinants
# for use with slate
"""I don;t belive we are going to implement slater but these are here
if needed"""
#  flags.backflow = False
#  flags.build_backflow = False

# Jastrow factor Flags
"""more explanation on what this is???"""

flags.jastrow_en = False
flags.jastrow_ee = False
flags.jastrow_een = False

"""There are flags for debugging, do we need these?"""


#  -----------------------------
"""These are the original values of the input values"""


original_flags.multi_gpu = False  # number of GPU's used
original_flags.batch_size = 10   # number of walkers
original_flags.double_precision = False  # the double or single precision
original_flags.pretrain_iterations = 1000  # the iterations over which the network is pretrained
original_flags.pretrain_basis = 'sto-3g'  # To do with PySCF and using the Hartree-Fock calculation

# optimisation original_flags
original_flags.iterations = 100000  # number of iterations
original_flags.clip_el = 5.0  # the scale at with to clip local energy????
original_flags.learning_rate = 1e-4  # learning rate
original_flags.learning_rate_decay = 1.0  # exponent of learning rate decay
                                          # can be changed to include exponential i.e. 1E1
original_flags.learning_rate_delay = 10000.0  # the scale of the rate of decay

# original_flags for KFAC
"""KFAC isn't implemented at the moment but if it is
these are the parameters, ignore if not using KFAC"""
"""Need to impliment inital original_flags for ADAM is using that, must talk to JB"""

original_flags.use_kfac = False
original_flags.kfac_invert_every = 1
original_flags.kfac_cov_update_every = 1
original_flags.kfac_damping = 0.001
original_flags.kfac_coov_ema_decay = 0.95
original_flags.kfac_momentum_type = 'regular'
original_flags.kfac_adapt_damping = False
original_flags.kfac_damping_adaption_decay = 0.9
original_flags.kfac_damping_adaptation_interval = 5
original_flags.kfac_min_damping = 1e-4
original_flags._kfac_norm_contraint = 0.001

# original_flags for system

original_flags.system_charge = 0  # we will only be working with atoms so this will always be 0
original_flags.system_dim = 3  # dimensions of the system
# hydrogen circle, are we going to be implementing any hydrogen circles???
original_flags.system_type = 'molecule'  # shouldn't this be an input value? ... unclear (may not need this flag)
original_flags.system = "H"     # We will only be working with atoms so use the atomic symbol
                                # if we use molecules then the system should be the molecular symbol
                                # if "hn" the number of atoms in the hydrogen chain
original_flags.units = "bohr"  # the input units of the system. Check with everyone to see what our system uses

"""are we going to be using diatomics, hydrogen chains and hydrogen circles
if we are then these are the original_flags relating to these systems, if not simply ignore"""

original_flags.system_separation = 0.0  # for the hydrogen chain and diatomics, this is the separation between nuclei
                                        # for Hydrogen circles, this is the radius of the circle
                                        # for diatomics this will default to the bond length when 0
original_flags.system_angle = np.pi/4.0  # Angle from the x-axis for the hydrogen circle


#  original_flags for logging
"""This is to do with saving and logging results. More original_flags maybe
rrequired depending on how this is implemented"""
original_flags.deterministic = False
original_flags.result_folder = abspath(join(dirname(__file__), 'results'))  # where and what the results folder is located and named
original_flags.stats_frequency = 1  # iterations between stat logging
original_flags.save_frequency = 10.0  # time (minutes??) between saving network parameters
original_flags.log_walkers = False  # whether to log walker values after every iteration. Comes with high data usage warning
original_flags.log_local_energies = False  # whether to log all local energies for each walker at each step
original_flags.log_wavefunction = False  # the same as above but for the wavefunction instead
original_flags.random_seed = 1


#  the name of the file where we save command line arguments
original_flags.cmdline_file = 'command_line_arguments.txt'


# original_flags for MCMC chain

original_flags.mcmc_burn_in = 100  # number of burn in steps after pretraining
                                   # if 0 will not burn in or reinitialise walkers
original_flags.mcmc_steps_per_update = 10  # number of mcmc steps between each update
original_flags.mcmc_initial_width = 0.8  # Gaussian used to generate the initial electron configuration
original_flags.mcmc_move_width = 0.2  # Width of the Gaussian used for the random moves
original_flags.mcmc_initial_offset = None  # was in Neil's code, must ask if this is inherited or needs initialisation
original_flags.mcmc_initial_stddev = None  # was in Neil's code, must ask if this is inherited or needs initialisation
original_flags.mcmc_init_means = ''  # If left empty, the electrons are assigned to atoms based on isolated spin configuration
                                     # The initial configurations of the the 3*nelectrons and their positions

# original_flags for Will's Pseudocode
"""Must ask Will how many of these are actually required
 and what values would be suitable
there may be crossover with some of the original_flags already
 present so only some alterations would be required"""
original_flags.fnn_electron_positions = None
original_flags.fnn_nuclei_positions = None
original_flags.fnn_n_up = None
original_flags.fnn_L = None
original_flags.fermilayer_h_in_dims = None
original_flags.fermilayer_h_out_dims = None


# original_flags for Andrew's Pseudocode
"""Must ask Andrew how many of these are actually required
 and what values would be suitable
there may be crossover with some of the original_flags already
 present so only some alterations would be required"""
original_flags.atoms = None
original_flags.nelectrons = None
original_flags.potential_epsilon = None
original_flags.x = None
original_flags.xs = None


# original_flags for the network architecture

original_flags.network_architecture = 'ferminet'  # we will be running ferminet exclusively I believe but here
                                                  # is where different networks would be chosen
original_flags.hidden_units = '((32, 4), (32, 4), (32, 4), (32, 4))'  # The number of hidden units in each layer of the network
                                                                      # For Ferminet, the first number in each tuple is the number
                                                                      # units in the 1-electron stream and the second number is
                                                                      # the units in the 2 electron stream
original_flags.determinants = 16  # number of determinents in the ferminet
original_flags.r12_distance_between_particle = True  # include r12/distance features between electrons and nuclei
original_flags.r12_distance_between_electrons = True  # same as above but between electron pairs
original_flags.pos_ee_features = True  # Include electron-electron position features
original_flags.use_envelope = True  # include multiplicative exponentially decaying envelope
original_flags.residual = True  # Use residual connections
original_flags.after_det = 1  # Coma-separated configuration of neural network after the determinants
# for use with slate
"""I don;t belive we are going to implement slater but these are here
if needed"""
# original_flags.backflow = False
# original_flags.build_backflow = False

# Jastrow factor original_flags
"""more explanation on what this is???"""

original_flags.jastrow_en = False
original_flags.jastrow_ee = False
original_flags.jastrow_een = False

"""There are original_flags for debugging, do we need these?"""
