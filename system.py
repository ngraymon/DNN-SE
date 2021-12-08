""" system module

This module provides classes and functions related to simulating
the quantum mechanical aspect of the systems; such as
atoms, charge, and bond lengths.

"""

# system imports
from typing import Sequence
from collections import namedtuple

# third party imports
import numpy as np

# local imports
import elements
# import units

"""
Default spin polarisation for a few diatomics of interest.
Otherwise default to either singlet (doublet) for even (odd) numbers of
electrons. Units: number of unpaired electrons.
"""
diatomic_spin_polarisation = {
    'B2': 2,
    'O2': 2,
    'NH': 2,
    'AlN': 2,
}

# the parent class of the `Atom` class
_atom_namedtuple = namedtuple(typename='atom', field_names=['symbol', 'number', 'coords', 'charge'])


class Atom(_atom_namedtuple):
    """ Atom information for Hamiltonians.
        Implemented as a subclass of the `namedtuple` `_atom_namedtuple`.

        Might need to be changed to a `SimpleNamespace` if mutability is important
        in the future.

        The nuclear charge is inferred from the symbol if not given, in which case the
        symbol must be the IUPAC symbol of the desired element.

        Attributes:
        symbol: Element symbol.
        coords: An iterable of atomic coordinates. Always a list of floats and in
            bohr after initialisation.
            Default: place atom at origin.
        charge: Nuclear charge.
            Default: nuclear charge (atomic number) of atom of the given name.
        atomic_number: Atomic number associated with element.
            Default: atomic number of element of the given symbol.
            Should match charge unless fractional nuclear charges are being used.
        units: String giving units of coords. Either bohr or angstrom.
            Default: bohr.
            If angstrom, coords are converted to be in bohr and units to the string 'bohr'.
        coords_angstrom: list of atomic coordinates in angstrom.
        coords_array: Numpy array of atomic coordinates in bohr.
        element: elements.Element corresponding to the symbol.
    """

    def build_basic_atom(symbol, spins=None, charge=0):
        """ Create a single atom.

        Args:
            symbol: The atomic symbol from the periodic table
            spins (optional): A tuple with the number of spin-up and spin-down electrons
            charge (optional): If zero (default), create a neutral atom, otherwise
              create an anion if charge is negative or cation if charge is positive.
        Returns:
            A list with a single Atom object located at zero, and a tuple with the spin
            configuration of the electrons.
        """

        # the `elements` module should implement a helper function for this
        number = 1 if symbol == 'H' else elements.SYMBOLS[symbol].atomic_number

        if charge > number:
            raise ValueError(
                "Cannot have a cation with charge larger than the atomic number. "
                f"Charge: {charge}, Atomic Number {number}"
            )

        if spins is None:
            # the `elements` module should implement a helper function for this
            # spin_polarization = elements.ATOMIC_NUMS[number-charge].spin_config

            if symbol == 'H':
                spin_polarization = 0
            else:
                raise Exception('need to fully implement the `elements` module')

            nof_alpha = (number + spin_polarization) // 2
            nof_beta = number - charge - nof_alpha

            # the number of spins in the (up, down) directions
            spin_config = (nof_alpha, nof_beta)

        atom = Atom(symbol, number, coords=(0.0, 0.0, 0.0), charge=number)

        return [atom, ], spin_config


def hydrogen_chains(n, width, charge=0):
    """Return a hydrogen chain with `n` atoms and separation `r`."""
    nof_electrons = n - charge

    # if even number of electrons then take half up-spin, half down-spin
    if nof_electrons % 2 == 0:
        spin_config = (nof_electrons // 2, nof_electrons // 2)
    # otherwise we prefer more up-spin electrons
    else:
        spin_config = ((nof_electrons+1) // 2, (nof_electrons-1) // 2)

    z_lim = 0.5 * width * (nof_electrons - 1)

    atom_list = [
        Atom(symbol='H', number=1, coords=(0.0, 0.0, z), charge=1)
        for z in np.linspace(-z_lim, z_lim, n)
    ]

    return atom_list, spin_config


def methane():
    atom_list = [
        Atom(symbol='C', number=6, coords=(0.0, 0.0, 0.0), charge=6),
        Atom(symbol='H', number=1, coords=(1.18886, 1.18886, 1.18886), charge=1),
        Atom(symbol='H', number=1, coords=(-1.18886, -1.18886, 1.18886), charge=1),
        Atom(symbol='H', number=1, coords=(1.18886, -1.18886, -1.18886), charge=1),
        Atom(symbol='H', number=1, coords=(-1.18886, 1.18886, -1.18886), charge=1),
    ]
    spin_config = (5, 5)

    return atom_list, spin_config
