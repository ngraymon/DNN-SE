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
import numpy as np
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
            initial_offset: mean of normal distribution
                from which initial configurations are drawn
            initial_stddev: standard deviation of normal distribution
                from which initial configurations are drawn
            offset: mean of normal distribution for drawing mc moves
            stddev: standard deviation of normal distribution for drawing mc moves
        """

        # store pointer to the network object
        self.net = network

        # record our distribution configuration
        self._init_offset = initial_offset
        self._init_stddev = initial_stddev
        self._offset = offset
        self._stddev = stddev

        # initialize our walkers
        self.walker_shape = (1, batch_size)  # this probably needs to be (3*N, 1, batch_size)
        # self.walkers = np.zeros(shape=self.walker_shape, dtype=dtype)
        self.walkers = self._initial_random_states()

        # number of monte carlo steps to preform per epoch/network call
        self.nof_steps = nof_steps

        # how we analyze our mc progress
        self.psi = self.compute_psi()
        self.rolling_accuracy = 0.0

        return

    def compute_psi(self):
        """ x """
        return self.net.forward(self.walkers)

    def pre_train(HF_orbitals, nof_steps, **kwargs):
        """
        I vaguely remember that they pre-trained on HF orbitals...
        we might need functions for that?
        """
        return

    def compute_probability_density_of_state(self, state, *args):
        """ Compute the probability density of g()

        this will depend on how we define:
        - our Hamiltonian
        - a state
        - wavefunction forms
        etc

        """
        dim_of_samples = args[0]

        # extract the wavefunction/probability density function (constructed by the network)
        # this is our P() function
        # pdf = args

        result = rng.random(
            loc=self._offset,
            scale=self._stddev,
            size=dim_of_samples
        )

        return result

    def compute_probability_ratio(self, cur_state, new_state, *args):
        """Compute the probability ratio between
        the proposed sample x` and the previous sample x_t.
        """

        g_of_new_state = self.compute_probability_density_of_state(new_state, *args)
        log.debug(f"{'P(x`)':<30}{g_of_new_state[0]:.8f}")

        g_of_current_state = self.compute_probability_density_of_state(cur_state, *args)
        log.debug(f"{'P(x_t)':<30}{g_of_current_state[0]:.8f}")

        try:
            probability_ratio = g_of_new_state / g_of_current_state
            log.debug(f"{'probability_ratio':<30}{probability_ratio[0]:.8f}")
        except Exception as e:
            print("numerical issue")
            raise e

        return probability_ratio

    def compute_proposal_density(self, state, *args, proposal_density='gaussian'):
        """ For now we just use a gaussian as the proposal density.
        This should probably get changed in the future?
        """

        # assuming this is the appropriate treatment a.t.m
        # also assume that we have multidimensional arrays?
        x = state
        mu, sigma = args  # a.k.a [loc, scale]

        prefactor = np.prod(np.sqrt(1.0 / (2 * np.pi * sigma**2)))

        exponential = -0.5 * (np.power(x - mu, 2.0) / (2 * sigma**2))

        g_of_x = prefactor * np.exp(exponential)

        return g_of_x

    def compute_proposal_density_ratio(cur_state, new_state, *args):
        """Compute ratio of the proposal density in two directions.
        From x_t to x` and conversely from x` to x_t.
        This is equal to 1 if the proposal density is symmetric.
        """
        dim_of_samples, mu, sigma = args

        rho_of_new_state = compute_proposal_density(cur_state, mu, sigma)
        log.debug(f"{'g(x_t | x`)':<30}{rho_of_new_state[0]:.8f}")

        rho_of_current_state = compute_proposal_density(new_state, mu, sigma)
        log.debug(f"{'g(x` | x_t)':<30}{rho_of_current_state[0]:.8f}")

        proposal_density_ratio = rho_of_current_state / rho_of_new_state
        log.debug(f"{'proposal_density_ratio':<30}{proposal_density_ratio[0]:.8f}")

        return proposal_density_ratio

    def compute_acceptance_ratio(cur_state, new_state, *args):
        """ This function calculates the acceptance_ratio
        `alpha = f(x')/f(x_t)`
        for a proposed new state `x'`
        given the current state `x_t`
        """

        # calculate a_1 = P(x') / P(x)
        probability_ratio = compute_probability_ratio(cur_state, new_state, *args)

        log_small_horizontal_line()

        # calculate a_2 = g(x|x') / g(x'|x)
        proposal_density_ratio = compute_proposal_density_ratio(cur_state, new_state, *args)

        # calculate min(1, a_1 * a_2)
        acceptance_ratio = min(np.ones_like(probability_ratio), probability_ratio * proposal_density_ratio)

        log_small_horizontal_line()
        log.debug(f"{'acceptance_ratio':<30}{acceptance_ratio:} = min(1.0, {probability_ratio * proposal_density_ratio:})")

        return acceptance_ratio

    def propose_new_state(self, dim_of_samples, *args):
        """ Generate a possibly new state.

        Return `dim_of_samples` from the distribution `pdf`

        see https://numpy.org/doc/stable/reference/random/generator.html
        for more distribution options:
        - multivariate_normal(mean, cov[, size, …])
        - lognormal([mean, sigma, size])

        do we need to care about the envelop at all? if so it seems we
        would have to turn to scipy; something like `scipy.stats.truncnorm` from
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        """

        # mu, sigma = args

        # for now we assume normal
        delta = rng.normal(
            loc=self._offset,
            scale=self._stddev,
            size=self.walker_shape
        )

        new_state = self.walkers + delta

        return new_state

    def _initial_random_states(self):
        """ Compute initial walker points """

        state = rng.normal(
            loc=self._init_offset,
            scale=self._init_stddev,
            size=self.walker_shape
        )
        return state

    def metropolis_accept_step(self, acceptance_ratio, walkers):
        """ This function evaluates the 'proposed' new_state
        and returns `True` if it is accepted otherwise 'False'
        """

        # generate our uniform random number
        u = np.random.uniform(size=self.walkers.shape)
        log.debug(f"{'uniform random number':<30}{u[0]:.8f}")

        # i have to call forward to generate my new phis
        new_phi = self.net.forward(walkers)[0]

        # test the condition
        accepted = bool(u <= acceptance_ratio)
        log.debug(f"{accepted=}")

        return accepted, new_phi

    def preform_one_step(
        self,
        list_of_states,
        list_of_psi,
        list_of_ratios,
        list_of_accepts,
        dim_of_samples, *args, **kwargs
    ):
        """ Using a standard Metropolis-Hastings algorithm
        see https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
        """

        # 1 - some input parameters that the step depends on
        cur_state = list_of_states[-1]
        cur_psi = list_of_psi[-1]
        accuracy = self.rolling_accuracy

        # 2 - draw a new step and wavefunction
        new_state = self.propose_new_state(dim_of_samples, *args)
        new_psi = self.net.forward(new_state)

        # 3 - compute acceptance ratio
        acceptance_ratio = np.squeeze(2 * (new_psi - cur_psi))
        # acceptance_ratio = compute_acceptance_ratio(cur_state, new_state, dim_of_samples, *args)
        list_of_ratios.append(acceptance_ratio[0])

        # 4 - calculate if we accept or reject this step
        accepted_bools, new_phi = self.metropolis_accept_step(acceptance_ratio, )
        list_of_accepts.append(accepted_bools)

        # need to change this to broadcasted acceptance because accepted will be N-d array
        cur_state = np.where(accepted_bools, new_state, cur_state)
        cur_psi = np.where(accepted_bools, new_psi, cur_psi)

        # 3 - update relevant objects/parameters
        list_of_states.append(cur_state)
        list_of_psi.append(cur_psi)
        self.rolling_accuracy = accuracy = np.mean(accepted_bools.astype(f32))

        return cur_state, cur_psi, accuracy

    def print_sorted_ratios(list_of_ratios):
        values = {}
        for n in list_of_ratios:
            if n in values:
                values[n] += 1
            else:
                values[n] = 1

        for k, v in values.items():
            print(f"The ratio {k: >30} occurred {v: >6} times")

        return

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
        dtype=f32,
    )

    # instantiate objects for storage
    original_state = mcmc.propose_new_state(dim_of_samples, *args)
    list_of_states = [original_state, ]
    list_of_ratios, list_of_accepts = [], []

    log.debug(f"Preparing to preform {mcmc.nof_steps} monte carlo steps")

    # preform a few steps to produce log output
    for i in range(0, flags.mcmc_steps_per_update):
        log_large_horizontal_line(i)
        mcmc.preform_one_step(list_of_states, list_of_ratios, list_of_accepts, dim_of_samples, *args)
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
