'''
Hamiltonian module

This module contains the functions for constructing Hamiltonians and setting 
up the VMC calculations.

Import this module as:
    import Hamiltonian as Ham
'''

# Import
import numpy as np
import torch


def kinetic_from_log(f,x):
    r"""Compute -1/2 \nable^2 \psi / \psi from log|psi|."""
    
    
def operators(atoms, nelectrons, potential_epsilon=0.0):
    
    Nucleus to Nucleus Potential = 0.0   
    
    FOR EACH ATOM I IN THE SYSTEM:
        FOR each atom in the system after atom I:
            CHARGE = ATOM1 CHARGE * ATOM2 CHARGE
            POTENTIAL = CHARGE/(DISTANCE BETWEEN ATOMS)
            ADD potential to Nucleus to Nuclesu Potential
            
    def smooth_norm(x):
        
        IF the potential_epsilon is 0:
            RETURN x
        ELSE:
            RETURN the square root of the sum of x squared plus potential_epsilon squared.
    
    
    def nuclear_potential(xs):
        
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
