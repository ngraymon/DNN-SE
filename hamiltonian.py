'''
Hamiltonian module

This module contains the functions for constructing Hamiltonians and setting 
up the VMC calculations.

Import this module as:
    import Hamiltonian as H
'''

# Import
import numpy as np
import torch


def kinetic_from_log(f,x):
    '''
    Computes the kinetic energy from the log of |psi|, 
    the -1/2 \nabla^2 \psi / \psi.

    Parameters
    ----------
    f : Torch Tensor
        Tensor for the wavefunction components
    x : Torch Tensor
        Tensor for the coordinates

    Returns
    -------
    The kinetic energy function.

    '''
    
    df = torch.autograd(f,x)[0]
    for i in range(len(f))
        
    
    
def operators(atoms, nelectrons, potential_epsilon=0.0):
    '''
    Creates the kinetic and potential operators of the Hamiltonian in atomic 
    units.

    Parameters
    ----------
    atoms : List of objects
        A list of objects from the Atom class.
    nelectrons : Integer
        The number of electrons in the system.
    potential_epsilon : Argument
        Epsilon used to smooth the divergence of the 1/r potential near the
        origin. The default is 0.0.

    Returns
    -------
    Functions for the kinetic and potential energy as a pytorch operator
    '''
    
            
    def smooth_norm(x):
        '''
        Function used to smooth out an instabilities when x approaches 0 in 
        functions involving 1/x.
        
        Parameters
        ----------
        x : Torch tensor of float points
          Values that approaches 0
        
        Returns
        -------
        The norm of the tensors rows.
        '''
        
        # If their is no instability then return the norm of x
        if potential_epsilon == 0: 
            return torch.norm(x,dim=1,keepdim=True)
        # Else we add the epsilon term then return the norm.
        else: 
            return torch.sqrt(torch.sum(x**2 + potential_epsilon**2, 
                                        dim=1,keepdim=True))
    
    
    def nuclear_potential(e_positions):
        '''
        Calculates the nuclear potential for set of electron positions.

        Parameters
        ----------
        e_positions : Torch tensor
            A tensor of electron positions.

        Returns
        -------
        The potential between the nuclues and the electrons.
        '''
        
        # the potental for each nucleus
        v = []
        # Add up all the potentials between all the nucleus and their electorns
        for atom in atoms:
            charge = torch.tensor(atom.charge, dtype = e_positions[0].dtype)
            coords = torch.tensor(atom.coord, dtype = e_positions[0].dtype)
            v.extend([-charge / smooth_norm(coords - x) for x in e_positions])
        v = torch.tensor(v)
        return torch.sum(v)
    
    
    def electronic_potential(e_positions):
        '''
        Calculates the electric potential for the set of electron positions.

        Parameters
        ----------
        e_positions : Torch tensor
            A tensor of electron positions.

        Returns
        -------
        The potential between the electrons.
        '''
        
        # If there is more the one electron in the system.
        if len(e_positions) > 1:
            v = []
            for (i,ri) in enumerate(e_positions):
                v.extend([1/ smooth_norm(ri - rj) for rj in xs[i + 1:]])
            v = torch.tensor(v)
            return torch.sum(v)
        else:
            return torch.tensor(0.0)
  
    
    def nuclear_nuclear(dtype):
        '''
        Calculates the potential between all the nucleus' in the system.

        Parameters
        ----------
        dtype : Torch Type
            The type of the tensor to be returned.

        Returns
        -------
        Torch Tensor for the potential of the nucleus'.
        '''
        
        # The nucleus to nucleus potential
        vnn = 0.0   
    
        # Loops over all the combinations of atoms in the system
        for i, atom_i in enumerate(atoms):
            for atom_j in atomes[i+1:]:
                # Charge of atom i an atom j.
                qij = float(atom_i.charge * atom_j.charge)
                # Add the potential between atom i and atom j.
                vnn += qij / np.linalg.norm(atom_i.coords_array 
                                            - atom_j.coords_array)

        return torch.tensor([vnn],dtype = dtype)
        
    
    def potential(positions):
        '''
        Splits the tensor x into the tensor xs for the electron positions. 
        Then compute the potntials and adds them together to return the total
        potential.

        Parameters
        ----------
        positions : Torch Tensor
            The position tensor for the electrons and nucleus'.

        Returns
        -------
        The total potential
        '''
        
        e_positions = torch.split(positions,nelectrons,dim=1)
        
        return (nuclear_potential(e_positions) 
              + electronic_potential(e_positions)
              + nuclear_nuclear(e_positions.dtype))
        



def exact_hamiltonian(atoms, nelectrons, potential_epsilon = 0.0):
    '''
    Evaluates the exact hamiltonian of a system.

    Parameters
    ----------
    atoms : Object
        The object that contains the atoms properties.
    nelectrons : Integer
        The number of electrons in the system.
    potential_epsilon : Float, optional
        Value to fix instability around 1/r. The default is 0.0.

    Returns
    -------
    The functions that generates the wavefunction and the hamiltonian op.
    '''
    
    # The kinetic and the potential functions.
    k_fn, v_fn = operators(atoms, nelectronsm potential_epsilon = 0.0)
    
    def _hamiltonian(f, x):
        logpsi, signpsi = f(x)
        psi = torch.exp(logpsi) * signpsi
        hpsi = psi * (k_fn(logpsi, x) + v_fn(x))
        return psi, hpsi
    
    return _hamiltonian




def r12_features(x, atoms, nelectrons, keep_pos=True, flatten=False,
                 atomic_coords=False):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    atoms : TYPE
        DESCRIPTION.
    nelectrons : TYPE
        DESCRIPTION.
    keep_pos : TYPE, optional
        DESCRIPTION. The default is True.
    flatten : TYPE, optional
        DESCRIPTION. The default is False.
    atomic_coords : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    