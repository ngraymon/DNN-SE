""" fermi_nerual_net module

This module contains the network and associated functions.
Import this module as follows:
    import fermi_neural_net as fnn

"""

# system imports

# third party imports
import torch.nn as nn
import numpy as np

# local imports


class FermiNeuralNet(nn.Module):
    """ Prototype network """

    def __init__(self):
        """ Initialization function """
        super(FermiNeuralNet, self).__init__()

    def forward(self, features):
        """ Feedforward function """

        # has tanh in it?

    def reset(self):
        """ Reset function for the training weights
        Use if the same network is trained multiple times.
        """

    def backprop(self, data, loss, epoch, optimizer):
        """ backpropagation function """

        self.train()

        features, labels = data

        # model makes a prediction
        prediction = self.forward(features)

        # compute instantaneous loss
        loss_object = loss(prediction, labels)

        # reset the gradients; to prevent double counting
        optimizer.zero_grad()

        # backpropagate the loss, calculates gradient values
        loss_object.backward()

        # adjust the hyperparameters using the gradient values
        optimizer.step()

        return loss_object.item()


def prepare_initial_h_tensors(walker_confs, nuclear_positions):
    """ Setup the tensors on line 2 & 3 in algorithm 1 """

    # note the the `walker_configuration` should always be sorted so that
    # all spin up electrons come first and all spin down electrons follow

    single_stream_h, double_stream_h = [], []

    for e1 in walker_confs:

        single_stream_h.append([
            [e1 - R, np.abs(e1 - R)] for R in nuclear_positions
        ])

        double_stream_h.append([
            [e1 - e2, np.abs(e1 - e2)] for e2 in walker_confs
        ])

    return single_stream_h, double_stream_h


def _rough_idea_of_computing_values_for_a_layer(h1, h2, *args):
    """ I think this function will be split up when the network is defined,
    but i find it helpful to lay out the function to get an idea of the shape
    of the tensors."""

    n_up, n_down, nof_electrons, V, W, bias_1, bias_2 = args

    # compute the averages
    g1_up = np.mean(h1[0:n_up])
    g1_down = np.mean(h1[n_up+1:])

    for i in range(0, n_up):
        g2_up = np.mean(h2[i])

    for i in range(n_up+1, n_up+n_down):
        g2_down = np.mean(h2[i])

    f_tensor = np.concatenate(h1, g1_up, g1_down, g2_up, g2_down)

    new_single_h = h1 + np.tanh((V * f_tensor) + bias_1)
    new_double_h = h2 + np.tanh((W * h2) + bias_2)

    return new_single_h, new_double_h


def compute_envelop(k, i, j, *args):
    """ Compute the second term from equation 6 """
    weights, covariance_matrx, walker_confs, nuclear_positions = args

    # have to check dimensionality, its unclear
    exponent = -np.abs(covariance_matrx[k, i] * (walker_confs[j] - nuclear_positions))

    # it seems like we trace out the m'th dimension (# of nuclear positions)
    ret = np.sum(weights[k, i] * np.exp(exponent), axis=3)
    return ret


def compute_orbital(k, i, j, envelop, h2_final, *args):
    """ Compute the orbitals expressed in equation 6 """

    W, n_up, n_down, g2_up, g2_down = args

    if j < n_up:
        output = np.dot(W[k, i], h2_final[j]) + g2_up[k, i]
    elif j >= n_up:
        output = np.dot(W[k, i], h2_final[j]) + g2_down[k, i]
    else:
        raise Exception()

    dampened_output = output * envelop

    return dampened_output


def compute_determinants(walker_confs, h2_final, *args):
    """ x """

    # unpack
    nof_det, nof_orbitals, det_weights, n_up = args
    nof_electrons = len(walker_confs)

    # just use lists for now
    det_up, det_down = [], []

    orbital = np.zeros((nof_orbitals, nof_electrons), dtype=np.float32)

    # compute the determinants
    for k in range(nof_det):
        for i in range(nof_orbitals):
            up_spin_orbitals = bool(i < n_up)
            for j in range(nof_electrons):
                up_spin_electrons = bool(j < n_up)

                # if the spin of the orbitals and electrons are opposite then the orbital is zero
                if up_spin_orbitals != up_spin_electrons:
                    shape = None  # i am not sure what shape this is
                    orbital[i, j] = np.zeros(shape, dtype=np.float32)
                else:
                    envelop = compute_envelop(k, i, j, *args)
                    orbital[i, j] = compute_orbital(k, i, j, envelop, h2_final, *args)

        det_up.append(np.linalg.det(orbital[0:n_up]))
        det_down.append(np.linalg.det(orbital[n_up+1:]))

        # might need to use np.linalg.slogdet() instead?
        # see https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet

    return np.array(det_up), np.array(det_down)


def evaluate_algorithm(walker_confs, nuclear_positions, nof_layers, *args):
    """ just a rough outline """

    h1, h2 = prepare_initial_h_tensors(walker_confs, nuclear_positions)
    single_stream_h = [h1, ]
    double_stream_h = [h2, ]

    # process the layers
    for l in range(nof_layers):

        h1, h2 = _rough_idea_of_computing_values_for_a_layer(h1, h2, *args)
        single_stream_h.append(h1)
        double_stream_h.append(h2)

    # so at this point the only thing we pass on is the last double stream h
    # this is h^{L,alpha}_{j}
    h2_final = double_stream_h[-1]

    det_up, det_down = compute_determinants(walker_confs, h2_final, *args[1:])

    # assume that each of these arrays are only 1 dimensional
    det_weights = args[0]
    wavefunction = np.sum(det_weights * det_up * det_down)

    return wavefunction