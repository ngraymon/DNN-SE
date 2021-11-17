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
    r"""Compute -1/2 \nable^2 \psi / \psi from log|psi|."""
    df = torch.autograd(f,x)[0]
    
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
    
    # The nucleus to nucleus potential
    vnn = 0.0   
    
    # Loops over all the combinations of atoms in the system
    for i, atom_i in enumerate(atoms):
        for atom_j in atomes[i+1:]:
            # Charge of atom i an atom j.
            qij = float(atom_i.charge * atom_j.charge)
            # Add the potential between atom i and atom j.
            vnn += qij / np.linalg.norm(atom_i.coords_array - atom_j.coords_array)

            
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
    
    
    def nuclear_potential(xs):
        '''
        Calculates the nuclear potential for set of electron positions.

        Parameters
        ----------
        xs : Torch tensor
            A tensor of electron positions.

        Returns
        -------
        The potential between the nuclues and the electrons.
        '''
        
        # the potental for each nucleus
        v = []
        
        CREATE an empty list for electron nucleus potential 
        
        FOR atom in atoms:
            ADD (-charge/ smooth_norm(coord - x) for x in sx) to the potential list 
        RETURN the sum of the potential lists for all the atom in atoms
    
    
    def electronic_potential(xs):
        
        IF there is more then one electron
            CREATE an empty list for electron electron potential
            For every electron in the system
                APPEND the inverse of the smooth_norm of the distance between this electron and all the other electrons
            RETURN the sum of the electron electron potentials
        ELSE 
            RETURN 0.0
    
    
    def nuclear_nuclear(dtype):
        RETURN the Nucleus to Nucleus potential 
        
    
    def potential(x):
        SPLIT x into nelectrons arrays
        
        RETURN the sum of the nuclear_potential, the electronic_potential, and the nuclear_nuclear
    
    RETURN the kinetic_from_log, potential
    

<<<<<<< Updated upstream
=======

def exact_hamiltonian(atoms, nelectrons, potential_epsilon = 0.0):
    
    CALL on the operator function and returns the kinetic_from_log and potential_epsilon
    
    def _hamiltonian(f,x):
        
        GET the log of psi and the sign of psi 
        
        PSI is the exponent of psi times the sign of psi
        THE Hamiltonian is the kinetic_from_log plus the potential all multiplied by psi
        
        RETURN psi and the hamamiltonian
        
    RETURN _hamiltonian

def
    
>>>>>>> Stashed changes
