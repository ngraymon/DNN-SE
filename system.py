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


def group_number(symbol, number, period):
    """ return the group number of an atom """

    is_lanthanide = (58 <= number <= 71)
    is_actinide = (90 <= number <= 103)

    if is_lanthanide or is_actinide:
        return -1
    if symbol == 'He':
        return 18

    period_starts = (1, 3, 11, 19, 37, 55, 87)
    period_start = period_starts[period - 1]
    group_ = number - period_start + 1

    # Adjust for absence of d block in periods 2 and 3.
    if period < 4 and group_ > 2:
        group_ += 10

    # Adjust for Lanthanides and Actinides in periods 6 and 7.
    if period >= 6 and group_ > 3:
        group_ -= 14

    return group_


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
        if symbol == 'H':
            number = 1
        else:
            for d in elements.e_list:
                if symbol == d['symbol']:
                    number = d['atomic_number']
                    break

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
                unpaired_e_dict = {1: 1, 2: 0, 13: 1, 14: 2, 15: 3, 16: 2, 17: 1, 18: 0}
                d = elements.e_list[number - 1 - charge]
                gn = group_number(symbol, number, d['period'])
                spin_polarization = unpaired_e_dict[gn]

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


def diatomic(symbol1, symbol2, bond_length, spins=None, charge=0, units='bohr'):
    """Return configuration for a diatomic molecule."""
    if spins is None:

        for d in elements.e_list:
            if symbol1 == d['symbol']:
                number_1 = d['atomic_number']
            if symbol2 == d['symbol']:
                number_2 = d['atomic_number']
        else:
            print('fail')

        assert isinstance(number_1, int)
        assert isinstance(number_2, int)

        total_charge = number_1 + number_2 - charge

        if total_charge % 2 == 0:
            spins = (total_charge // 2, total_charge // 2)
        else:
            spins = ((total_charge + 1)// 2, (total_charge - 1) // 2)

    return [
      Atom(symbol=symbol1, number=number_1, coords=(0.0, 0.0, bond_length/2.0), charge=number_1),
      Atom(symbol=symbol2, number=number_2, coords=(0.0, 0.0, -bond_length/2.0), charge=number_2)
    ], spins


def helium():
    atom_list = [
        Atom(symbol='He', number=2, coords=(0.0, 0.0, 0.0), charge=0),
    ]
    spin_config = (1, 1)

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


def ammonia():
    atom_list = [
        Atom(symbol='N', number=7, coords=(0.0, 0.0, 0.22013), charge=7),
        Atom(symbol='H', number=1, coords=(0.0, 1.77583, -0.51364), charge=1),
        Atom(symbol='H', number=1, coords=(1.53791, -0.88791, -0.51364), charge=1),
        Atom(symbol='H', number=1, coords=(-1.53791, -0.88791, -0.51364), charge=1),
    ]
    spin_config = (5, 5)

    return atom_list, spin_config


def ethylene():
    atom_list = [
        Atom(symbol='C', number=6, coords=(0.0, 0.0, 1.26135), charge=6),
        Atom(symbol='C', number=6, coords=(0.0, 0.0, -1.26135), charge=6),
        Atom(symbol='H', number=1, coords=(0.0, 1.74390, 2.33889), charge=1),
        Atom(symbol='H', number=1, coords=(0.0, -1.74390, 2.33889), charge=1),
        Atom(symbol='H', number=1, coords=(0.0, 1.74390, -2.33889), charge=1),
        Atom(symbol='H', number=1, coords=(0.0, -1.74390, -2.33889), charge=1),
    ]
    spin_config = (8, 8)

    return atom_list, spin_config


def bicyclobutane():
    atom_list = [
        Atom(symbol='C', number=6, coords=(0.0, 2.13792, 0.58661), charge=6),
        Atom(symbol='C', number=6, coords=(0.0, -2.13792, 0.58661), charge=6),
        Atom(symbol='C', number=6, coords=(1.41342, 0.0, -0.58924), charge=6),
        Atom(symbol='C', number=6, coords=(-1.41342, 0.0, -0.58924), charge=6),
        Atom(symbol='H', number=1, coords=(0.0, 2.33765, 2.64110), charge=1),
        Atom(symbol='H', number=1, coords=(0.0, 3.92566, -0.43023), charge=1),
        Atom(symbol='H', number=1, coords=(0.0, -2.33765, 2.64110), charge=1),
        Atom(symbol='H', number=1, coords=(0.0, -3.92566, -0.43023), charge=1),
        Atom(symbol='H', number=1, coords=(2.67285, 0.0, -2.19514), charge=1),
        Atom(symbol='H', number=1, coords=(-2.67285, 0.0, -2.19514), charge=1),
    ]
    spin_config = (15, 15)

    return atom_list, spin_config


def h4_circle(r, theta, units='bohr'):
    """Return 4 hydrogen atoms arranged in a circle, a failure case of CCSD(T)."""
    atom_list = [
        Atom(symbol='H', number=1, coords=(r*np.cos(theta), r*np.sin(theta), 0.0), charge=1),
        Atom(symbol='H', number=1, coords=(-r*np.cos(theta), r*np.sin(theta), 0.0), charge=1),
        Atom(symbol='H', number=1, coords=(r*np.cos(theta), -r*np.sin(theta), 0.0), charge=1),
        Atom(symbol='H', number=1, coords=(-r*np.cos(theta), -r*np.sin(theta), 0.0), charge=1)
    ]
    spin_config = (2, 2)

    return atom_list, spin_config
