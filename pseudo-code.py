import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.autograd import Function



class KFAC(Optimizer):
    """class of KFAC (approximate of NGD) here it is a preconditioner, so similar to optimizer but doesn't update parameters
    only changes the gradient of those param (with the fischer info) to apply it use:
    preconditioner.step()
    optimizer.step()
"""
    def __init__(self, NN,block_approx, tensor_approx,frequency ):
        """ NN (torch.nn.Module): Neural Network
            block and tensor approx (bool): wheater or not we apply these 2 approx
            frequency (int): the minibatch frequency at which we inverse the fischer info
        """

        #initialise the variables
        self.block_approx=block_approx
        self.tensor_approx=tensor_approx
        self.frequency=frequency
        self.counter=0 #running count of the number of itt

        """ create a way to extract the forward and backward sensitive layers to calculate Fischer
        """
        

        for mod in NN.modules():
            #over the linear elements of the NN
            mod_class=mod.__class__.__name__
            #extract the layer names of the class
            if mod_class in ['Linear', 'Conv2d']:
            # create a way to store the weights, the mod in question and the layer type 
                params=[mod.weight]
                # add weights if they exist

                all_param={'params': params, 'mod':mod, 'layer_type':mod_class}
                self.params.append(all_param)
            super(KFAC,self).__init__(self.params, {})


        """ parameters of super class optimizer: self.state contains all the information about the weights and the shape of the NN
            self.param_group allows to easily itterate around the different layers
        """
    

    def apply(self):
        """ applies the inverse matrix to the weights and updates the gradients of the weight
        """
        for layers in self.param_groups:
            weights=layers['params'][0]
            state=self.state[weights]  #layers['params'][1] is biases
        
        self.compute_fischer_block(layers,state)
        self.inv_F(state['xxt'],state['ggt'])
        #update the gradients of weights and biases with KFAC
        weights.grad.data=self.update_g(weights,biases,layers,state)




    def compute_fischer_block(self,layers,state):
        """ we update in state the values of xxt and ggt, with x being the activation from the previous layer 
        and g being the the back propagated gradient  """

        mod=layers['mod']
        x=self.state[mod]['x']
        g=self.state[mod]['gy']

        ##compute xxt (i.e x. xtranspose)
        with torch.no_grad(): #we dont want to change the gradients of the weights to compute this
            state['xxt']= 1# something here see paper how they normalise for the wave function

            state['ggt']= 1#something 

    
    def inv_F(xxt,ggt,state):
      with torch.no_grad():
        state['ixxt']=1#something
        state['iggt']=1#something
      
    def update_g(weight,biases,layers,state):
        with torch.no_grad():
            g=weight.grad
            s=g.shape
            invxxt = state['ixxt']
            invggt = state['iggt']
            g=torch.mm(torch.mm(invggt,g),invxxt)
            
            g.contiguous().view(*s)


def my_loss(psi):
    #fake loss function to caculate the grad of psi
    return psi

def updateweights(hamiltonian,model,walkers,KFAC,lr):
    psi=model.forward(walkers)
    kinetic_fn, potential_fn = hamiltonian
    kinetic = kinetic_fn(psi, walkers)
    potential = potential_fn(walkers)
    local_energy = kinetic + potential
    loss =torch.mean(local_energy)

    mean_loss=torch.mean(loss) #different kind of mean?
    grad_loss=local_energy-mean_loss

    #calculates the gradient of psi (logpsi) by taking the autograd grad of the loss
    #but instead of giving it the loss we give it the psi output layer for a specific x
    psiloss=my_loss(psi)
    optimizer=torch.optim.SGD(psiloss,lr)
    psiloss.backward()
    #apply the precondtioner
    KFAC.step()
    
    for p in model.parameters():
        #update the grad of psi with (E_local-mean(E_local))
        p.grad()*=grad_loss
    optimizer.step()


    #now the grad of the loss is caculated we want to precondition the local
    #gradients of the linear part of the 



