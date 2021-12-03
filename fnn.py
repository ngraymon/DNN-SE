""" fnn module

This module defines the neural network.

"""

# system imports
import itertools as it
from os import walk
from os.path import join, abspath

# third party imports
import torch
import numpy as np

# local imports


class FermiNet(torch.nn.Module):
    """ x """

    def __init__(self, spin_config, electron_positions, nuclei_positions, hidden_units, num_determinants=10):
        """ x """
        super(FermiNet, self).__init__()

        # tuple of size 2 such as (4, 6): 4 spin up, 6 spin down
        self.spin_config = spin_config
        self.n_up, self.n_down = n_up, n_down = spin_config
        self.n = n = sum(spin_config)
        self.I = I = len(nuclei_positions)
        self.nuclei_positions = nuclei_positions
        self.num_determinants = num_determinants

        self.n_layers = len(hidden_units)
        h_sizes = [[4*I, 4]] + hidden_units
        h_sizes[-1][1] = 0  # layers that are not connected have a weight of zero

        self._initialize_network_layers(n, n_up, h_sizes)

        self.preprocess(electron_positions)

        self._initialize_trainable_parameters(num_determinants, n, I, h_sizes)

    def _initialize_network_layers(self, n, n_up, layer_dims):
        """ Create the networks layers

        n - number of electrons
        n_up - number of up spin electrons
        layer_dims - list of tuples (a, b) where a is the size of the single streams
            and b is the size of the double streams
        """
        self.layers = [
            FermiLayer(n, n_up, layer_dims[i], layer_dims[i+1])
            for i in range(self.n_layers)
        ]

    def _initialize_trainable_parameters(self, nof_determinants, nof_electrons, nof_nuclei, h_sizes):
        """ Randomly initialize trainable parameters

        The `h_sizes[-1][0]` is the last layer's zero'th ????
        """

        # lambda function takes any number of args and passes them on
        p_func = lambda *size: torch.nn.Parameter(torch.rand(size))

        # w vectors have size (nof_determinants)
        self.final_weights = p_func(nof_determinants, nof_electrons, h_sizes[-1][0])

        # g scalars
        self.final_biases = p_func(nof_determinants, nof_electrons)

        # pi scalars for decaying envelopes
        self.pi_weights = p_func(nof_determinants, nof_electrons, nof_nuclei)

        # sigma scalars for decaying envelopes
        self.sigma_weights = p_func(nof_determinants, nof_electrons, nof_nuclei)

        # omega scalars for summing determinants
        self.omega_weights = p_func(nof_determinants)

    def _preprocess_single_stream(self, electron_positions):
        """ Create the torch tensor storing the distance between electrons and nuclei
        see https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        """

        # create empty tensor
        shape = (self.n, self.I, 3)
        eN_vectors = torch.empty(shape)

        """ the broadcasting for this operation is as follows
        (A, B, C) = (A, 1, C) - (1, B, C)
            - C is always 3 (representing the three cartesian co-ordinates)
            - A is the # of electrons
            - B is the number of nuclei

        so for methane with 5 atoms and 10 electrons you get
            (10, 5, 3) = (10, 1, 3) - (1, 5, 3)
        """
        print(f"{electron_positions.shape = }")
        print(f"{self.nuclei_positions.shape = }")

        eN_vectors[:] = torch.unsqueeze(electron_positions, 1) - torch.unsqueeze(self.nuclei_positions, 0)
        print(f"{eN_vectors.shape = }")

        """ if we have (10, 5, 3) then take the norm along the 3 dimension (cartesian co-ordinates)
        and get a (10, 5, 1) output
        see (https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) for more info
        """
        norm_eN_vectors = torch.linalg.vector_norm(eN_vectors, dim=2, keepdim=True)

        """ see function `_build` from line 729 in `networks.py` for reference
        its unclear if we need some of this code yet?
        print(f"{eN_vectors.shape = }\n{self.spin_config}")
        a = torch.split(eN_vectors, self.spin_config, dim=0)
        """

        # concatenate (10, 5, 3) with (10, 5, 1) to get (10, 5, 4)
        single_stream = torch.concat((eN_vectors, norm_eN_vectors), dim=2)
        print(f"{single_stream.shape = }")

        if False:
            # if instead you want to end up with (10, 20)
            # where 20 is 5*3 positions concatenated with 5*1 norms
            single_stream = torch.concat((eN_vectors.flatten(1), norm_eN_vectors.flatten(1)), dim=1)
            print(f"{single_stream.shape = }")

        return single_stream

    def _preprocess_double_stream(self, electron_positions):
        """ Create the torch tensor storing the distance between pairs of electrons """

        # create empty tensor
        shape = (self.n, self.n, 3)
        ee_vectors = torch.empty(shape)

        """ the broadcasting for this operation is as follows
        (A, B, C) = (A, 1, C) - (1, B, C)
            - C is always 3 (representing the three cartesian co-ordinates)
            - A is the # of electrons
            - B is the # of electrons

        so for methane with 10 electrons you get
            (10, 10, 3) = (10, 1, 3) - (1, 10, 3)
        """
        ee_vectors[:] = torch.unsqueeze(electron_positions, 1) - torch.unsqueeze(electron_positions, 0)
        print(f"{ee_vectors.shape = }")

        """ if we have (10, 10, 3) then take the norm along the 3 dimension (cartesian co-ordinates)
        and get a (10, 10, 1) output
        see (https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) for more info
        """
        norm_ee_vectors = torch.linalg.vector_norm(ee_vectors, dim=2, keepdim=True)

        # concatenate (10, 10, 3) with (10, 10, 1) to get (10, 10, 4)
        double_stream = torch.concat((ee_vectors, norm_ee_vectors), dim=2)
        print(f"{double_stream.shape = }")

        if False:
            # if instead you want to end up with (10, 40)
            # where 20 is 10*3 positions concatenated with 10*1 norms
            double_stream = torch.concat((ee_vectors.flatten(1), norm_ee_vectors.flatten(1)), dim=1)
            print(f"{double_stream.shape = }")

        return double_stream

    def preprocess(self, electron_positions):
        """ Prepare the visible layers from the single and double steam h tensors """

        if torch.is_tensor(electron_positions):
            electron_positions = electron_positions.detach()

        print(type(self.nuclei_positions), self.nuclei_positions.shape)
        print(type(electron_positions), electron_positions.shape)

        # single_h_vecs_vector
        single_h_stream = self._preprocess_single_stream(electron_positions)

        # double_h_vecs_matrix
        double_h_stream = self._preprocess_double_stream(electron_positions)

        # debug
        print(f"{single_h_stream.shape = }")
        print(f"{double_h_stream.shape = }")

        # concatenate the two streams
        self.visible_layers = [single_h_stream, double_h_stream]

    # saves the model to a file
    def save(self, path):
        # this first one saves the whole network, everything included (and dependencies)
        torch.save(self, path)
        # this one only saves the weights and biases (i.e. tuneable parameters)
        torch.save(self.state_dict(), join(path, '_state_dict'))
        print("FermiNet model saved to file '{0}'. To load, use 'model = torch.load({0})'".format(path))

    ### 'electron_positions' and 'walker' are aliases, for readability
    ### If walker is given, that is used as the electron_positions instead
    ### This functionality isn't necessary, it's to help if people don't realise the walkers are just electron positions
    def forward(self, electron_positions, multi=False):
        """ x """

        print(type(electron_positions))
        import pdb; pdb.set_trace()

        if electron_positions is not None:
            self.preprocess(electron_positions)

        # if multi is True, then the network forwards a list of walkers and returns a list of outputs
        # this should just be replaced with broadcasting?!
        if multi:
            return [self.forward(electron_positions=electron_positions[i]) for i in range(len(electron_positions))]

        layer_outputs = [self.visible_layers]
        for layer in self.layers[:-1]:
            layer_outputs.append(layer.layer_forward(layer_outputs[-1], self.n_up))

        layer_outputs.append(self.layers[-1].layer_forward(layer_outputs[-1], self.n_up))

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

    def __init__(self, n, n_up, h_in_dims, h_out_dims):
        """ x """
        super().__init__()

        if (n_up == 0) or (n-n_up == 0):
            # case with all electrons having same spin (i.e. no up spins or no down spins)
            f_vector_length = 2*h_in_dims[0] + 1*h_in_dims[1]
        else:
            f_vector_length = 3*h_in_dims[0] + 2*h_in_dims[1]

        # lambda function takes any number of args and passes them on
        p_func = lambda *size: torch.nn.Parameter(torch.rand(size))

        """ matrix and bias vector for each single stream's linear op applied to the f vector
            and yielding a vector of the output length

        self.v_matrices is a vector of matrices
        self.b_vectors is a vector of vectors
        """
        self.v_matrices = p_func(n, f_vector_length, h_out_dims[0])
        self.b_vectors = p_func(n, h_out_dims[0])

        """ matrix and bias vector for each double stream's linear op applied to the f vector
            and yielding a vector of the output length

        self.w_matrices is a matrix of matrices
        self.c_vectors is a matrix of vectors
        """
        self.w_matrices = p_func(n, n, h_in_dims[1], h_out_dims[1])
        self.c_vectors = p_func(n, n, h_out_dims[1])

    def layer_forward(self, input_tensor, n_up):
        """ x """

        # single layers:
        single_h, double_h = input_tensor[0].type(torch.FloatTensor), input_tensor[1].type(torch.FloatTensor)
        single_h_up, single_h_down = single_h[:n_up], single_h[n_up:]
        double_h_ups, double_h_downs = double_h[:, :n_up], double_h[:, n_up:]

        n = len(input_tensor[0])

        single_g_up = torch.mean(single_h_up, 0) if single_h_up.nelement() else torch.empty(0)
        single_g_down = torch.mean(single_h_down, 0) if single_h_down.nelement() else torch.empty(0)
        double_g_ups = torch.mean(double_h_ups, 1, keepdim=True) if double_h_ups.nelement() else torch.empty(n, 0)
        double_g_downs = torch.mean(double_h_downs, 1, keepdim=True) if double_h_downs.nelement() else torch.empty(n, 0)

        if False:  # debug
            print('\n')
            print(f"{single_h[0].shape = }")
            print(f"{single_g_up.shape = }")
            print(f"{single_g_down.shape = }")
            print(f"{double_g_ups[0].shape = }")
            print(f"{double_g_downs[0].shape = }")
            print('\n')

        f_vectors = torch.stack([
            torch.cat((
                single_h[i],
                single_g_up,
                single_g_down,
                double_g_ups[i],
                double_g_downs[i]
            )) for i in range(n)]).type(torch.FloatTensor)

        """ f_vectors is size (10, 17, 1, 4) for methane
        17 is:
            - 5 for single_h
            - 5 for single_g_up
            - 5 for single_g_down
            - 1 for double_g_ups (because it was a mean)
            - 1 for double_g_downs (because it was a mean)
        """

        """ torch.transpose(self.v_matrices, 1, 2) is size (10, 32, 68) for methan
        """

        # single_output = torch.tanh(torch.bmm(torch.transpose(self.v_matrices, 1, 2), f_vectors[:,:,None]) + self.b_vectors)  # Note: check dimensions order for torch.mul are correct??

        import pdb; pdb.set_trace()

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
