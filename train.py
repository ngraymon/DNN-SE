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
from log_conf import log
from flags import flags


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

        # stores the loss for each epoch
        loss_array = []
        # phi_phisgn = [[], ]

        # creating the walkers...
        for i in range(self.param['epoch']):

            self.net.zero_grad()
            # get wavefunction for each one of these configuration, creates the configurations for each electron
            # for a given batch size
            phi, walkers, accuracy = self.mcmc.preform_one_step()

            # copy_of_phi = torch.tensor(phi)  # make sure the grad of phi is not changed in `self.kinetic`

            assert not torch.any(torch.isnan(walkers)), 'state configuration is borked'

            # assert phi.require_grad is True, 'fuck'

            # from the Hamiltonian extract potential and kinetic energy
            kinetic_value = self.kinetic(phi, walkers, self.net)
            potential_value = self.potential(walkers)

            log.debug(f"{kinetic_value.shape = }")
            log.debug(f"{potential_value.shape = }")
            # import pdb; pdb.set_trace()
            local_energy = kinetic_value + potential_value
            log.debug(f"{local_energy.shape = }")

            # this is the "real" loss of the system, i.e the mean of the loss per batch
            loss = torch.mean(local_energy)

            # take the mean over the GPU replicas
            if flags.multi_gpu:
                mean_gpu_loss = torch.mean(loss, axis=0)
            else:
                # if only 1 GPU then just keep the loss
                mean_gpu_loss = loss

            # default for now
            if self.clip_el is None:
                # here is the loss being passed into the backward pass since we have an explicit
                # expression for the gradient of the loss
                relative_term = local_energy - mean_gpu_loss
                # log.debug(f"{relative_term = }")
                # log.debug(f"{copy_of_phi = }")
                # log.debug(f"{relative_term.shape = }")
                # log.debug(f"{copy_of_phi.shape = }")
                computed_loss = torch.mean(relative_term * phi)

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
            self.optimizer.step()
            loss_array.append(loss)
            # phi_phisgn.append()

            log.debug(f"{loss_array = }")
            log.debug(f"Completed epoch {i+1}")
            import pdb; pdb.set_trace()

        return loss_array
