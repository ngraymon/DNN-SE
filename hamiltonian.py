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
    
    POTENTIAL = 0.0   
    
    FOR EACH ATOM I IN THE SYSTEM:
        FOR each atom in the system after atom I:
            CHARGE = ATOM1 CHARGE * ATOM2 CHARGE
            POTENTIAL = CHARGE/(DISTANCE BETWEEN ATOMS)
            
            
    def smooth_norm(x):
        
        IF the potential_epsilon is 0:
            RETURN x
        ELSE:
            RETURN the square root of the sum of x squared plus potential_epsilon squared.
    
    
    def nuclear_potential(xs):
        
        CREATE an empty list for potential
        
        FOR atom in atoms:
            ADD (-charge/ smooth_norm(coord - x) for x in sx) to the potential list 
        RETURN the sum of the potential lists for all the atom in atoms
    
    
        