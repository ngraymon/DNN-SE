""" train module

This module preforms the training

"""

# system imports

# third party imports
import torch
from torch import optim
from torch.autograd import grad_mode
from torch.optim import optimizer
from torch.optim.optimizer import Optimizer

# local imports


class Train():
    def __init__(self, network, mcmc, hamiltonian_operators, param, clip_el=None):
        """
        network: a call to the ferminet network class

        mcmc:an instance of the mcmc class allowing to create an object that samples configurations from
        psi^2

        Hamiltonian: a call to the class Hamiltonian to calculate the local energy

        param: hyperparameters
        """
        self.net = network
        self.mcmc = mcmc
        self.kinetic, self.potential = hamiltonian_operators
        self.param = param
        self.optimizer = optim.Adam(self.net.parameters(), param['lr'])
        self.clip_el = clip_el  # the factor applied to distance for clipping local energy

    def train_KFAC(self):
        return 0

    def train(self, bool_KFAC=False, clipping=False):
        if bool_KFAC:
            self.train_KFAC()

        losstot = []
        phi_phisgn = [[], ]

        # creating the walkers...
        for i in range(self.param['epoch']):

            self.net.zero_grad()
            # get wavefunction for each one of these configuration, creates the configurations for each electron
            # for a given batch size
            phi, walkers, accuracy = self.mcmc.preform_one_step()

            copy_of_phi = torch.tensor(phi)  # make sure the grad of phi is not changed in `self.kinetic`

            assert not torch.any(torch.isnan(walkers)), 'state configuration is borked'

            # from the Hamiltonian extract potential and kinetic energy
            kinetic = self.kinetic(phi, walkers, self.net)
            potential = self.potential(walkers)
            local_energy = kinetic + potential

            # this is the "real" loss of the system, i.e the mean of the loss for that batch size
            loss = torch.mean(local_energy, axis=1)

            # default for now
            if self.clip_el is None:
                # here is the loss being passed into the backward pass since we have an explicit
                # expression for the gradient of the loss
                computed_loss = torch.mean((local_energy - loss) * copy_of_phi)

            else:

                #
                median = torch.median(local_energy)

                #
                diff = torch.mean(torch.abs(local_energy - median))

                #
                cliped_local = torch.clip(
                    local_energy,
                    median - self.clip_el*diff,
                    median + self.clip_el*diff
                )

                #
                computed_loss = torch.mean((cliped_local - loss) * copy_of_phi)

            # compute the gradient w.r.t. the weights and update with ADAM
            computed_loss.backward()
            Optimizer.step()
            losstot.append(loss)
            # phi_phisgn.append()

        return losstot
