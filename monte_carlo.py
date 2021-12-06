""" monte_carlo module

This module contains all functions related to computing a quantum mechanical wavefunction
by imaginary time propagation (monte carlo integral approximation)
Import this module as follows:
    import monte_carlo as mc


You can run it with
    python3 monte_carlo.py

"""

# system imports
from os.path import abspath, join, dirname

# third party imports
import torch
import numpy as np
from numpy import newaxis as NEW
from numpy.random import default_rng
from numpy import float32 as f32


# local imports
import log_conf
from log_conf import log, log_small_horizontal_line, log_large_horizontal_line
from flags import flags


# initialize the random number generator
rng = default_rng()


output_dir = abspath(join(dirname(__file__)))


def print_small_horizontal_line():
    print("-"*40)


def print_large_horizontal_line(index=None):
    """Prints a header if index is an integer, otherwise
    prints a horizontal line of length 60.
    """
    if index is not None:
        assert isinstance(index, int), f"index parameter must be an integer not a {type(index)}"
        string = f" State {index:>3d} "
        print(f"{'':-^60}\n{string:-^60}\n{'':-^60}")
    else:
        print("-"*60)


class MonteCarlo():
    """ Handles the Metropolis-Hastings Monte Carlo operations.

    Memeber functions store the current variables related to MC such as:
        - `walkers`: positions of electrons
        - 'psi':  wavefunction
    """

    def __init__(
        self,
        network,
        batch_size,
        initial_offset, initial_stddev,
        offset, stddev,
        nof_steps,
        dtype=f32,
    ):
        """Creates `MonteCarlo` object.

        We draw the initial configurations with offset and standard deviation
        provided by `initial_offset` and `initial_stddev`.
        But for each step beyond the initial conifguration we draw new
        proposed samples using `offset` and `stddev`

        Args:
            batch_size: number of configurations to genereate
            initial_offset: list of means of normal distribution from which initial
                configurations are drawn, of length 3*N where N is the number of electrons.
            initial_stddev: standard deviation of normal distribution
                from which initial configurations are drawn
            offset: mean of normal distribution for drawing mc moves
            stddev: standard deviation of normal distribution for drawing mc moves
        """

        # store pointer to the network object
        self.net = network

        # record our distribution configuration
        self._init_offset = torch.tensor(initial_offset)
        self._init_stddev = initial_stddev
        self._offset = offset
        self._stddev = stddev

        # initialize our walkers
        # nof_electrons = len(initial_offset) // 3

        # the empty dimension `1` at the end is necessary for correct concatenation
        # when sampling
        # self.walker_shape = (batch_size, len(initial_offset))
        self.batch_size = batch_size
        # self.walkers = np.zeros(shape=self.walker_shape, dtype=dtype)

        number_of_replicas = 1  # how many GPUs we are using

        shape = (number_of_replicas, batch_size)
        self.walkers = self._initial_random_states(shape)
        self.net_multi = bool(len(self.walkers.shape) >= 3)

        # number of monte carlo steps to preform per epoch/network call
        self.nof_steps = nof_steps

        # how we analyze our mc progress
        log.debug(f"{self.walkers.shape = }")
        self.psi = self.compute_psi(self.walkers)
        log.debug(f"{self.psi.shape = }")

        assert not torch.isnan(self.psi), 'Initial wavefunction is nan'
        self.rolling_accuracy = 0.0

        return

    def compute_psi(self, visible_nodes, *args, **kwargs):
        """ x """
        kwargs.update({'multi': self.net_multi})
        return self.net.forward(visible_nodes, *args, **kwargs)

    def pre_train(HF_orbitals, nof_steps, **kwargs):
        """
        I vaguely remember that they pre-trained on HF orbitals...
        we might need functions for that?
        """
        return

    def propose_new_state(self):
        """ Generate a possibly new state.

        Return `walker_shape` samples from the normal distribution
        specified by `_offset` and `_stddev`.

        see https://numpy.org/doc/stable/reference/random/generator.html
        for more distribution options:
        - multivariate_normal(mean, cov[, size, …])
        - lognormal([mean, sigma, size])

        do we need to care about the envelop at all? if so it seems we
        would have to turn to scipy; something like `scipy.stats.truncnorm` from
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        """

        delta = np.array([
            rng.normal(loc=self._init_offset, scale=self._init_stddev,)
            for b in range(self.batch_size)
        ])

        new_state = self.walkers + torch.tensor(delta)

        return new_state

    def _initial_random_states(self, shape):
        """ Compute initial walker points.

        Return `self.walker_shape` samples from the normal distribution
        specified by the `_init_offset` and `_init_stddev`
        """

        log.debug(f"{self._init_offset.shape = }")
        log.debug(f"{self._init_stddev = }")

        states = np.array([
            rng.normal(loc=self._init_offset, scale=self._init_stddev,)
            for b in range(self.batch_size)
        ])

        log.debug(f"{states.shape = }")
        return torch.tensor(states, requires_grad=True)

    def metropolis_accept_step(self, acceptance_ratio):
        """ This function evaluates the 'proposed' new_state
        and returns `True` if it is accepted otherwise 'False'
        """

        # generate our uniform random number
        u = torch.rand(size=self.walkers.shape)
        log.debug(f"{'uniform random number at index 0':<30}{u[0, ...]}")

        # test the condition
        accepted = u <= acceptance_ratio
        log.debug(f"{accepted = }")

        return accepted

    def preform_one_step(self, *args, **kwargs):
        """ Using a standard Metropolis-Hastings algorithm
        see https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
        """
        record_steps = kwargs.get("record_steps", False)

        if record_steps:
            list_of_states, list_of_psi, list_of_ratios, list_of_accepts = args

        # 1 - some input parameters that the step depends on
        cur_state = self.walkers
        cur_psi = self.psi
        accuracy = self.rolling_accuracy

        # OKAY! so for sure I can do the following and it works!
        # BUT it looks like new_state is not part of the network!?!?
        # df = torch.autograd.grad(self.psi, self.walkers)

        # 2 - draw a new step and wavefunction
        new_state = self.propose_new_state()
        # here ferminet seems to take the zeroth element of the array
        new_psi = self.compute_psi(new_state)
        assert not torch.isnan(new_psi), 'New psi is nan'

        # 3 - compute acceptance ratio
        acceptance_ratio = torch.squeeze(2 * (new_psi - cur_psi))
        log.debug(f"{acceptance_ratio.shape = }")

        if record_steps:
            list_of_ratios.append(acceptance_ratio[0])

        # 4 - calculate if we accept or reject for each step
        accepted_bools = self.metropolis_accept_step(acceptance_ratio)

        if record_steps:
            list_of_accepts.append(accepted_bools)

        # need to change this to broadcasted acceptance because accepted will be N-d array

        """ 5 - pick up states/psi's given decision array `accepted_bools`
        For each element in `accepted_bools` if it is:
          - `True` we set the element of `cur_state` to the corresponding element of `new_state`
          - `False` we set the element of `cur_state` to the corresponding element of `cur_state`

        The same process is followed for `cur_psi` and `new_psi`.
        """
        cur_state = torch.where(accepted_bools, new_state, cur_state)
        cur_psi = torch.where(accepted_bools, new_psi, cur_psi)

        # if we are storing our progress for analysis
        if record_steps:
            list_of_states.append(cur_state)
            list_of_psi.append(cur_psi)

        # 6 - update relevant objects/parameters
        self.walkers.data = cur_state
        self.psi.data = cur_psi
        self.rolling_accuracy = accuracy = torch.mean(accepted_bools.float())

        # df = grad(self.psi, self.walkers, allow_unused=True)
        # df = torch.autograd.grad(self.psi, self.walkers, grad_outputs=torch.ones_like(self.psi))

        # log.debug(f"{df = }")
        # log.debug(f"{df.shape = }")
        log.debug(f"{self.walkers.shape = }")
        log.debug(f"{self.psi.shape = }")

        return self.psi, self.walkers, accuracy

    def print_sorted_ratios(list_of_ratios):
        """ Debug/Profiling tool to investigate the distribution of the acceptace ratios. """
        values = {}
        for n in list_of_ratios:
            if n in values:
                values[n] += 1
            else:
                values[n] = 1

        for k, v in values.items():
            print(f"The ratio {k: >30} occurred {v: >6} times")

    def save_human_readable_x_values(state_array, nof_states, rows=1):
        """ Readable text for debugging """
        string = ""
        for s in range(0, nof_states+1):
            string += f" ------- State {s} -------"
            for r in range(0, rows):
                string += f"\nRow {r}\n"
                string += "  ".join([
                    "{:10.6f}".format(state_array[s][r])
                ])
            #
            string += "\n"

        path = join(output_dir, f'states_{nof_states}.txt')

        with open(path, 'w') as fp:
            fp.write(string)

        return


def test_template(fnn):
    """ Use for basic debugging of execution flow and tensor shapes. """

    log_conf.setLevelDebug()

    # some dummy parameters

    # use flag parameters to add this dimensionality
    # (3*n, nof_gpus, batch_size)

    dim_of_samples = (1, )
    args = mu, sigma = 0.0, 1.0

    # for this to work we need default **kwargs
    kwargs = {}
    mcmc = MonteCarlo(
        network=fnn.FermiNet(**kwargs),
        batch_size=flags.batch_size,
        initial_offset=flags.mcmc_initial_offset,
        initial_stddev=flags.mcmc_initial_stddev,
        offset=flags.mcmc_offset,
        stddev=flags.mcmc_stddev,
        nof_steps=flags.mcmc_steps_per_update,
        record_steps=True,
        dtype=f32,
    )

    # instantiate objects for storage
    original_state = mcmc.propose_new_state()
    list_of_states = [original_state, ]
    list_of_ratios, list_of_accepts = [], []

    log.debug(f"Preparing to preform {mcmc.nof_steps} monte carlo steps")

    # preform a few steps to produce log output
    for i in range(0, flags.mcmc_steps_per_update):
        log_large_horizontal_line(i)
        mcmc.preform_one_step(
            list_of_states,
            list_of_ratios,
            list_of_accepts,
            record_steps=True
        )

    # EOL

    state_array = np.array(list_of_states)

    print_large_horizontal_line()

    mcmc.print_sorted_ratios(list_of_ratios)
    print(list_of_accepts)
    print(f"Number of accepted moves {np.sum(list_of_accepts)}/{len(list_of_accepts)}")

    # save states to file
    mcmc.save_human_readable_x_values(state_array, mcmc.nof_steps, rows=1)
    return


if (__name__ == "__main__"):
    test_template()
