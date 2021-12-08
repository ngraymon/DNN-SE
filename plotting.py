
# system imports
import os
from os.path import join, abspath

# third party imports
import torch
import numpy as np

import matplotlib as mpl
# `use_backend` needed if no x-server is available
mpl.use(['pdf', 'Agg', 'svg'][1])
from matplotlib import pyplot as plt

# local imports
from flags import flags


def try_to_make_directory(results_path):
    if not os.path.exists(results_path):
        try:
            os.makedirs(results_path, exist_ok=True)
        except Exception as e:
            print(
                "It seems the results directory doesn't exist "
                "and I failed to create it with `os.makedirs`:\n"
                f"{abspath(results_path)}\n"
            )
            raise e


def plot_helium(root_path, network, use_latex=False):
    """ Plot the helium wavefunction over a range of electron positions """

    plt.rcParams['text.usetex'] = use_latex

    plotting_density = flags.plotting_density

    plot_path = join(root_path, flags.plot_path)

    try_to_make_directory(plot_path)  # create directory if it doesn't exist

    """ fix the first electron for the Helium atom and vary the second
    for example:
        (0.5, 0,0) a_0
        (x, 0, 0) a_0, with x=linespace(-1,1,20)
        (0.5 cosθ, 0.5 sinθ, 0) a_0
    """

    linear = np.array([
        [[0.5, 0, 0], [x, 0, 0]]
        for x in np.linspace(-1, 1, plotting_density)
    ])
    rotational = np.array([
        [[0.5, 0, 0], [0.5*np.cos(theta), 0.5*np.sin(theta), 0]]
        for theta in np.deg2rad(np.linspace(-np.pi, np.pi, plotting_density))
    ])

    phi_lin = network.forward(torch.tensor(linear)).detach()
    phi_rot = network.forward(torch.tensor(rotational)).detach()

    fig, ax = plt.subplots(2)
    fig.suptitle('Wave function for helium atom')
    fig.tight_layout()

    # Make your plot, set your axes labels

    ax[0].plot(torch.linspace(-1, 1, plotting_density), torch.exp(phi_lin))
    ax[0].set_xlabel('x [Bohr]')
    ax[0].set_ylabel(r' $\psi$')

    # Turn off tick labels
    ax[0].set_yticklabels([])

    ax[1].plot(torch.linspace(-np.pi, np.pi, plotting_density), torch.exp(phi_rot))
    ax[1].set_xlabel(r'$\theta$')
    ax[1].set_ylabel(r' $\psi$')
    ax[1].set_yticklabels([])

    strFile = join(plot_path, 'Wavefunction.png')
    print(strFile)

    # overwrite the previous file
    if os.path.isfile(strFile):
        os.remove(strFile)

    plt.savefig(strFile)


def plot_h4_circle(root_path, kinetic_fn, potential_fn, network, use_latex=False):
    """ Plot the h4 wavefunction over a range of bond angles """

    plt.rcParams['text.usetex'] = use_latex

    plotting_density = flags.plotting_density

    plot_path = join(root_path, flags.plot_path)

    try_to_make_directory(plot_path)  # create directory if it doesn't exist

    """ fix the first electron for the Helium atom and vary the second
    for example:
        (0.5, 0,0) a_0
        (x, 0, 0) a_0, with x=linespace(-1,1,20)
        (0.5 cosθ, 0.5 sinθ, 0) a_0
    """

    R = 3.2843  # bohr

    x_values = np.linspace(85.0, 95.0, plotting_density)

    positions = np.array([
        [
            [R*np.cos(theta), R*np.sin(theta), 0.0],
            [-R*np.cos(theta), R*np.sin(theta), 0.0],
            [R*np.cos(theta), -R*np.sin(theta), 0.0],
            [-R*np.cos(theta), -R*np.sin(theta), 0.0],
        ]
        for theta in np.deg2rad(x_values)
    ])

    positions = torch.tensor(positions, requires_grad=True)

    phi = network.forward(positions)
    print(phi.requires_grad)

    kinetic = kinetic_fn(phi, positions, network).detach()
    potential = potential_fn(positions).detach()

    local_energy = kinetic + potential  # what we want to minimize

    fig, ax = plt.subplots(1)
    # fig.suptitle('Wave function for helium atom')
    fig.tight_layout()

    # Make your plot, set your axes labels

    ax.plot(x_values, local_energy)
    ax.set_xlabel(r'$\theta (degrees)$')
    ax.set_ylabel(r' $Energy (a.u.)$')

    strFile = join(plot_path, 'Wavefunction.png')

    # overwrite the previous file
    if os.path.isfile(strFile):
        os.remove(strFile)

    plt.savefig(strFile)


def plot_loss(root_path, losstot, use_latex=False):
    """ Plot the loss per epoch """

    plt.rcParams['text.usetex'] = use_latex
    plot_path = join(root_path, flags.plot_path)
    try_to_make_directory(plot_path)  # create directory if it doesn't exist

    plt.plot(losstot)
    plt.xlabel('Epoch')
    plt.ylabel('Local Energy (loss) [J]')

    strFile = join(plot_path, 'Local_Energy.png')

    # overwrite the previous file
    if os.path.isfile(strFile):
        os.remove(strFile)

    plt.savefig(strFile)
