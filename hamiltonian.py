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
        FOR EACH ATOM IN THE SYSTEM AFTER ATOM I:
            CHARGE = ATOM1 CHARGE * ATOM2 CHARGE
            POTENTIAL = CHARGE/(DISTANCE BETWEEN ATOMS)
            
    