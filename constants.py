""" constants module

This module contains numerical definitions of any physical constants.

# references
https://aip.scitation.org/doi/10.1063/5.0064853
http://ws680.nist.gov/publication/get_pdf.cfm?pub_id=920687
http://ws680.nist.gov/publication/get_pdf.cfm?pub_id=920686

"""

# system imports

# third party imports
import numpy as np

# local imports

hbar = 1.0

# (joules / ev)
nist_j_per_ev = np.float64(1.6021766208e-19)

# (ev / kelvin)
boltzman = np.float64(1.38064852e-23) / nist_j_per_ev

# convert wavenumbers to electronVolts and back
wavenumber_per_eV = 8065.6  # 8065.54429
