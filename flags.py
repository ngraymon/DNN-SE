""" flags module
This module contains the definition of a SimpleNamespace object `flags` and initializes it with all the default values.
We're starting with a SimpleNamespace for simplicity.
https://docs.python.org/3/library/types.html#types.SimpleNamespace
We could also use namedtuples's:
https://docs.python.org/3/library/collections.html#collections.namedtuple

In FNN they use googles abseil, but starting with that package will probably present a roadblock to getting a MVP
so it is advisable to hold off for now.
https://abseil.io/
"""

# system imports
from types import SimpleNamespace

# third party imports

# local imports


""" simple instantiation
Use `flags` for testing and implementation
Make `original_flags` a direct copy of the FNN default flag settings for use in replicating their results.
"""
flags = SimpleNamespace()
original_flags = SimpleNamespace()


# -----------------------------
# start adding flags
flags.multi_gpu = False
flags.batch_size = 10
flags.double_precision = False
# flags.pretrain_iterations = 1000
# flags.pretrain_basis = 'sto-3g'



flags.mcmc_burn_in = 100
flags.mcmc_steps_per_update = 10
flags.mcmc_initial_width = 0.8
flags.mcmc_move_width = 0.2
# flags.mcmc_init_means = ''


flags.hidden_units = '((32, 4), (32, 4), (32, 4), (32, 4))'





# -----------------------------
original_flags.multi_gpu = False
original_flags.batch_size = 4096
original_flags.double_precision = False
original_flags.pretrain_iterations = 1000
original_flags.pretrain_basis = 'sto-3g'



original_flags.mcmc_burn_in = 100
original_flags.mcmc_steps_per_update = 10
original_flags.mcmc_initial_width = 0.8
original_flags.mcmc_move_width = 0.2
original_flags.mcmc_init_means = ''


original_flags.hidden_units = '((256, 32), (256, 32), (256, 32), (256, 32))'

