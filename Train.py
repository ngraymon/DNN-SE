import torch
from torch import optim
from torch.autograd import grad_mode
from torch.optim import optimizer
from torch.optim.optimizer import Optimizer


class Train():
    def __init__(self,network,mcmc,Hamiltonian,param):
        """
        network: a call to the ferminet network class

        mcmc:an instance of the mcmc class allowing to creat an object that samples configuariosn from 
        psi^2

        Hamiltonian: a call to the class hamiltonian to calculate the local energy

        param: hyperparameters
        """
        self.net=network
        self.mcmc=mcmc
        self.H=Hamiltonian
        self.param=param
        self.optimizer=optim.Adam(self.net.parameters(),param['lr'])


    def train_KFAC(self):
        return 0

  

    
    def train(self, bool_KFAC=False):
        if bool_KFAC:
            self.train_KFAC()
        
        losstot=[]
        phi_phisgn=[[]]
        for i in range(self.param['epoch']):
        #creating the walkers...
            walkers=self.mcmc.create() #creates the configurations for each electron
            #for a given batch size
            self.net.zero_grad()
            #get wavefunction for each one of these configuration
            phi, phi_sgn=self.net.forward(walkers)

            #from the hamiltonian extract potential and kinetic energy
            kinetic=self.H.kinetic(phi, walkers).detach()
            potential = self.H.potential(walkers).detach()
            local_energy = kinetic + potential

            # this is the "real" loss of the system
            loss=torch.mean(local_energy)
        
            #here is the loss being passed into the backward pass since we have an explicit
            #expression for the gradient of the loss
            computedloss=torch.mean((local_energy-loss)*phi)

            #compute the gradient wrt to weights and update with ADAM
            (computedloss).backward()
            Optimizer.step()
            losstot.append(loss)
            #phi_phisgn.append()
        return losstot

        


