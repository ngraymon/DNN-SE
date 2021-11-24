""" fnn module

This module defines the neural network.

"""

# system imports
import itertools as it
from os import walk

# third party imports
import torch
import numpy as np

# local imports


class FermiNet(torch.nn.Module):
    """ x """

    def __init__(self, L, n_up, electron_positions, nuclei_positions, custom_h_sizes=False, num_determinants=10):
        """ x """
        super(FermiNet, self).__init__()

        self.n, self.I = n, I = len(electron_positions), len(nuclei_positions)
        self.n_up = n_up
        self.num_determinants = num_determinants
        self.nuclei_positions = nuclei_positions

        """ default configuration, ensures correct dimensions in first layer, and keeps all h vectors the same length

        the final double stream layer is not necessary
        so setting `custom_h_sizes[-1]` to zero gives empty weight matrices and saves computations
        """
        if custom_h_sizes is False:
            custom_h_sizes = [[4*I, 4] for i in range(L+1)]
            custom_h_sizes[-1][1] = 0  # layers that are not connected have a weight of zero
        print(custom_h_sizes)

        self.layers = [FermiLayer(n, custom_h_sizes[i], custom_h_sizes[i+1]) for i in range(L)]

        self.preprocess(electron_positions)

        # Randomly initialise trainable parameters:
        self.final_weights = torch.nn.Parameter(torch.rand(self.num_determinants, n, custom_h_sizes[-1][0]))# w vectors
        self.final_biases = torch.nn.Parameter(torch.rand(self.num_determinants, n))  # g scalars
        self.pi_weights = torch.nn.Parameter(torch.rand(self.num_determinants, n, I)) # pi scalars for decaying envelopes
        self.sigma_weights = torch.nn.Parameter(torch.rand(self.num_determinants, n, I))  # sigma scalars for decaying envelopes
        self.omega_weights = torch.nn.Parameter(torch.rand(self.num_determinants))  # omega scalars for summing determinants


    def preprocess(self, electron_positions):
        # inputs in format [single_h_vecs_vector, double_h_vecs_matrix]
        self.inputs = [None, None]  # to be processed

        # single stream inputs:
        eN_vectors = np.array([[i-j for j in self.nuclei_positions] for i in electron_positions])
        self.eN_vectors = torch.from_numpy(eN_vectors)

        self.inputs[0] = torch.from_numpy(np.array([
            np.concatenate([
                np.concatenate(
                    [eN_vectors[i][j] for j in range(self.I)],
                    axis=None
                ),
                [np.linalg.norm(eN_vectors[i][j]) for j in range(self.I)]
            ])
            for i in range(self.n)
        ]))

        # double stream inputs:
        ee_vectors = [[i-j for j in electron_positions] for i in electron_positions]

        self.inputs[1] = torch.from_numpy(np.array([
            [
                np.concatenate([
                    ee_vectors[i][j],
                    [np.linalg.norm(ee_vectors[i][j])]
                ])
                for j in range(self.n)
            ]
            for i in range(self.n)
        ]))

    ### 'electron_positions' and 'walker' are aliases, for readability
    ### If walker is given, that is used as the electron_positions instead
    ### This functionality isn't necessary, it's to help if people don't realise the walkers are just electron positions
    def forward(self, electron_positions=None, walker=None, multi=False):
        """ x """

        if walker is not None: ################    walker is an alias of electron_positions
            electron_positions = walker ###    walker is an alias of electron_positions
        if electron_positions is not None:
            self.preprocess(electron_positions)

        # if multi is True, then the network forwards a list of walkers and returns a list of outputs
        if multi:
            return [self.forward(electron_positions=electron_positions[i]) for i in range(len(electron_positions))]


        layer_outputs = [self.inputs]
        counter = 0
        for i in self.layers[:-1]:
            counter += 1
            print(counter)
            layer_outputs.append(i.forward(layer_outputs[-1], self.n_up))

        layer_outputs.append(self.layers[-1].forward(layer_outputs[-1], self.n_up))

        # Compute final matrices:
        phi_up = torch.empty(self.num_determinants, self.n_up, self.n_up)
        phi_down = torch.empty(self.num_determinants, self.n - self.n_up, self.n - self.n_up)

        for k in range(self.num_determinants):

            # up spin:
            for i, j in it.product(range(self.n_up), repeat=2):
                final_dot = torch.dot(self.final_weights[k][i], layer_outputs[-1][0][j]) + self.final_biases[k][i]
                env_sum = torch.sum(torch.stack([self.pi_weights[k][i][m]*torch.exp(-torch.norm(self.sigma_weights[k][i][m] * self.eN_vectors[j][m])) for m in range(self.I)]))
                phi_up[k][i][j] = final_dot * env_sum

            # down spin:
            for i, j in it.product(range(self.n_up, self.n), repeat=2):
                final_dot = torch.dot(self.final_weights[k][i], layer_outputs[-1][0][j]) + self.final_biases[k][i]
                env_sum = torch.sum(torch.stack([self.pi_weights[k][i][m]*torch.exp(-torch.norm(self.sigma_weights[k][i][m] * self.eN_vectors[j][m])) for m in range(self.I)]))
                phi_down[k][i-self.n_up][j-self.n_up] = final_dot * env_sum

        # Compute determinants:
        d_up = torch.det(phi_up)
        d_down = torch.det(phi_down)

        # Weighted sum:
        wavefunction = torch.sum(self.omega_weights * d_up * d_down)  # sum of result of element-wise multiplications

        return wavefunction


class FermiLayer(torch.nn.Module):
    """ x """

    def __init__(self, n, h_in_dims, h_out_dims):
        """ x """
        super().__init__()

        f_vector_length = 3*h_in_dims[0] + 2*h_in_dims[1]

        """ matrix and bias vector for each single stream's linear op applied to the f vector
            and yielding a vector of the output length

        self.v_matrices is a vector of matrices
        self.b_vectors is a vector of vectors
        """
        self.v_matrices = torch.nn.Parameter(torch.rand(n, f_vector_length, h_out_dims[0]))
        self.b_vectors = torch.nn.Parameter(torch.rand(n, h_out_dims[0]))

        """ matrix and bias vector for each double stream's linear op applied to the f vector
            and yielding a vector of the output length

        self.w_matrices is a matrix of matrices
        self.c_vectors is a matrix of vectors
        """
        self.w_matrices = torch.nn.Parameter(torch.rand(n, n, h_in_dims[1], h_out_dims[1]))
        self.c_vectors = torch.nn.Parameter(torch.rand(n, n, h_out_dims[1]))


    def forward(self, input_tensor, n_up):
        """ x """

        # single layers:
        single_h, double_h = input_tensor[0].type(torch.FloatTensor), input_tensor[1].type(torch.FloatTensor)
        single_h_up, single_h_down = single_h[:n_up], single_h[n_up:]
        double_h_ups, double_h_downs = double_h[:, :n_up], double_h[:, n_up:]

        single_g_up = torch.mean(single_h_up, 0)
        single_g_down = torch.mean(single_h_down, 0)
        double_g_ups = torch.mean(double_h_ups, 1)  # Note: double check which axis?
        double_g_downs = torch.mean(double_h_downs, 1)  # Note: double check which axis?

        n = len(input_tensor[0])

        f_vectors = torch.stack([
            torch.cat((
                single_h[i],
                single_g_up,
                single_g_down,
                double_g_ups[i],
                double_g_downs[i]
            )) for i in range(n)]).type(torch.FloatTensor)

        # single_output = torch.tanh(torch.bmm(torch.transpose(self.v_matrices, 1, 2), f_vectors[:,:,None]) + self.b_vectors)  # Note: check dimensions order for torch.mul are correct??

        single_output = torch.tanh(
            torch.squeeze(
                # Note: check dimensions order for torch.mul are correct??
                torch.matmul(
                    torch.transpose(self.v_matrices, 1, 2),
                    f_vectors[:, :, None]
                ),
                dim=2
            ) + self.b_vectors
        )

        # output[0] = np.tanh(torch.tensor([(self.v_matrices[i] @ f_vectors[i]) + self.b_vectors[i] for i in range(len(f_vectors))]))
        if single_output.size() == single_h.size():
            single_output += single_h

        # double layers:
        # double_output = torch.tanh(torch.mul(self.w_matrices, double_h) + self.c_vectors)#Note: check dimensions order for @ (np.matmul) are correct??
        # Note: check dimensions order for torch.mul are correct??
        if self.w_matrices.size()[-1] == 0:
            w_mats_size = self.w_matrices.size()
            double_output = torch.zeros(w_mats_size[0]*w_mats_size[1], w_mats_size[-2], 1)
        else:
            double_output = torch.tanh(
                torch.squeeze(
                    torch.bmm(
                        torch.flatten(self.w_matrices, end_dim=1),
                        torch.flatten(double_h[:, :, :, None], end_dim=1)
                    ),
                    dim=2
                ) + torch.flatten(self.c_vectors, end_dim=1)
            )

        # reshape:
        shape = list(double_output.shape)[1:]
        double_output = double_output.reshape([n, n]+shape)
        # output[1] = np.tanh(torch.tensor([[(self.w_matrices[i][j] @ double_h[i][j]) + self.c_vectors[i][j] for j in range(len(double_h[0]))] for i in range(len(double_h))]))
        if double_output.size() == double_h.size():
            double_output += double_h

        return [single_output, double_output]
