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
from log_conf import log


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

        assert self.n > 0, 'must have at least 1 electron'  # just in case

        self.n_layers = len(hidden_units)

        # we have 4 = 3-(x,y,z)-co-ordinates + 1-vector_norm
        # the visible layer is (number_of_nuclei * 4, 4)
        # so for methane that would be ~ (5*4, 4) ~ (20, 4)
        self.layer_dimensions = [[I*4, 4], ]

        # then we add on the hidden layers
        self.layer_dimensions += hidden_units

        # set the output layer size of the last layer to zero
        # as layers that are not connected have a weight of zero
        self.layer_dimensions[-1][-1] = 0

        print(f"{self.layer_dimensions = }")
        # import pdb; pdb.set_trace()

        self._initialize_network_layers(spin_config, self.layer_dimensions)

        # we don't need to do this until we actually are processing input??
        # self.preprocess(electron_positions)

        self._initialize_trainable_parameters(num_determinants, n, I)

    def _initialize_network_layers(self, spin_config, layer_dims):
        """ Create the networks layers

        n - number of electrons
        n_up - number of up spin electrons
        layer_dims - list of tuples (a, b) where a is the size of the single streams
            and b is the size of the double streams
        """
        log.debug('Starting to initialize the network layers\n')
        self.layers = [
            FermiLayer(spin_config, (layer_dims[i], layer_dims[i+1]))
            for i in range(self.n_layers)
        ]
        log.debug('Finished initializing the network layers\n')

    def _initialize_trainable_parameters(self, nof_determinants, nof_electrons, nof_nuclei):
        """ Randomly initialize trainable parameters

        The `h_sizes[-1][0]` is the last layer's zero'th ????
        """

        # lambda function takes any number of args and passes them on
        p_func = lambda *size: torch.nn.Parameter(torch.rand(size))

        # w vectors have size (nof_determinants)
        self.final_weights = p_func(nof_determinants, nof_electrons, self.layer_dimensions[-1][0])

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
        self.eN_vectors = torch.empty(shape)

        """ the broadcasting for this operation is as follows
        (A, B, C) = (A, 1, C) - (1, B, C)
            - C is always 3 (representing the three cartesian co-ordinates)
            - A is the # of electrons
            - B is the number of nuclei

        so for methane with 5 atoms and 10 electrons you get
            (10, 5, 3) = (10, 1, 3) - (1, 5, 3)
        """
        log.debug(f"{electron_positions.shape = }")
        log.debug(f"{self.nuclei_positions.shape = }")

        self.eN_vectors = torch.unsqueeze(electron_positions, 1) - torch.unsqueeze(self.nuclei_positions, 0)
        log.debug(f"{self.eN_vectors.shape = }")

        """ if we have (10, 5, 3) then take the norm along the 3 dimension (cartesian co-ordinates)
        and get a (10, 5, 1) output
        see (https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) for more info
        """
        norm_eN_vectors = torch.linalg.vector_norm(self.eN_vectors, dim=2, keepdim=True)

        """ see function `_build` from line 729 in `networks.py` for reference
        its unclear if we need some of this code yet?
        log.debug(f"{eN_vectors.shape = }\n{self.spin_config}")
        a = torch.split(eN_vectors, self.spin_config, dim=0)
        """

        # concatenate (10, 5, 3) with (10, 5, 1) to get (10, 5, 4)
        single_stream = torch.cat((self.eN_vectors, norm_eN_vectors), dim=2)
        log.debug(f"{single_stream.shape = }")

        if False:
            # if instead you want to end up with (10, 20)
            # where 20 is 5*3 positions concatenated with 5*1 norms
            single_stream = torch.concat((self.eN_vectors.flatten(1), norm_eN_vectors.flatten(1)), dim=1)
            log.debug(f"{single_stream.shape = }")

        return single_stream

    def _preprocess_double_stream(self, electron_positions):
        """ Create the torch tensor storing the distance between pairs of electrons """

        # create empty tensor
        shape = (self.n, self.n, 3)
        self.ee_vectors = torch.empty(shape)

        """ the broadcasting for this operation is as follows
        (A, B, C) = (A, 1, C) - (1, B, C)
            - C is always 3 (representing the three cartesian co-ordinates)
            - A is the # of electrons
            - B is the # of electrons

        so for methane with 10 electrons you get
            (10, 10, 3) = (10, 1, 3) - (1, 10, 3)
        """
        self.ee_vectors = torch.unsqueeze(electron_positions, 1) - torch.unsqueeze(electron_positions, 0)
        log.debug(f"{self.ee_vectors.shape = }")

        """ if we have (10, 10, 3) then take the norm along the 3 dimension (cartesian co-ordinates)
        and get a (10, 10, 1) output
        see (https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) for more info
        """
        norm_ee_vectors = torch.linalg.vector_norm(self.ee_vectors, dim=2, keepdim=True)

        # concatenate (10, 10, 3) with (10, 10, 1) to get (10, 10, 4)
        double_stream = torch.cat((self.ee_vectors, norm_ee_vectors), dim=2)
        log.debug(f"{double_stream.shape = }")

        if False:
            # if instead you want to end up with (10, 40)
            # where 20 is 10*3 positions concatenated with 10*1 norms
            double_stream = torch.concat((self.ee_vectors.flatten(1), norm_ee_vectors.flatten(1)), dim=1)
            log.debug(f"{double_stream.shape = }")

        return double_stream

    def preprocess(self, electron_positions):
        """ Prepare the visible layers from the single and double steam h tensors """
        log.debug("Creating the single and double streams from the electron positions\n")

        log.debug((type(self.nuclei_positions), self.nuclei_positions.shape))
        log.debug((type(electron_positions), electron_positions.shape))

        # single_h_vecs_vector
        single_h_stream = self._preprocess_single_stream(electron_positions)
        log.debug(f"{single_h_stream = }")

        # double_h_vecs_matrix
        double_h_stream = self._preprocess_double_stream(electron_positions)
        log.debug(f"{double_h_stream = }")

        # debug
        log.debug(f"{single_h_stream.shape = }")
        log.debug(f"{double_h_stream.shape = }")

        """ Reminder that the two streams have different dimensionality
        and therefore we cannot simply concatenate them together.
        Using our prior examples the sizes are:
            single_h_stream: (10, 5, 4)
            double_h_stream: (10, 10, 4)

        Therefore we store them in a simple list
        """

        self.visible_layers = [single_h_stream, double_h_stream]
        log.debug("Finished creating the single and double streams\n")

    def save(self, path):
        """ saves the model to a file """

        # this first one saves the whole network, everything included (and dependencies)
        torch.save(self, path)

        # this one only saves the weights and biases (i.e. tune-able parameters)
        torch.save(self.state_dict(), join(path, '_state_dict'))

        log.debug("FermiNet model saved to file '{0}'. To load, use 'model = torch.load({0})'".format(path))

    def compute_up_spin_phis(self, layer_outputs, phi_up, k):
        """ up spin """

        # this can probably be improved to use broadcasting and torch calls
        for i, j in it.product(range(self.n_up), repeat=2):
            final_dot = self.final_biases[k][i] + torch.dot(
                self.final_weights[k][i],
                layer_outputs[-1][0][j]
            )
            env_sum = torch.sum(
                torch.stack(
                    [
                        self.pi_weights[k][i][m] * torch.exp(
                            -torch.norm(
                                self.sigma_weights[k][i][m] * self.eN_vectors[j][m]
                            )
                        )
                        for m in range(self.I)
                    ]
                )
            )

            phi_up[k][i][j] = final_dot * env_sum

    def compute_down_spin_phis(self, layer_outputs, phi_down, k):
        """ down spin """

        # this can probably be improved to use broadcasting and torch calls
        for i, j in it.product(range(self.n_up, self.n), repeat=2):
            final_dot = self.final_biases[k][i] + torch.dot(
                self.final_weights[k][i],
                layer_outputs[-1][0][j]
            )
            env_sum = torch.sum(
                torch.stack(
                    [
                        self.pi_weights[k][i][m] * torch.exp(
                            -torch.norm(
                                self.sigma_weights[k][i][m] * self.eN_vectors[j][m]
                            )
                        )
                        for m in range(self.I)
                    ]
                )
            )

            phi_down[k][i-self.n_up][j-self.n_up] = final_dot * env_sum

    def forward(self, electron_positions, multi=False):
        """ x """

        log.debug(type(electron_positions))
        # import pdb; pdb.set_trace()

        if electron_positions is not None:
            if not multi:
                self.preprocess(electron_positions)

        # if multi is True, then the network forwards a list of walkers and returns a list of outputs
        # this should just be replaced with broadcasting?!
        if multi:
            return [self.forward(electron_positions=electron_positions[i]) for i in range(len(electron_positions))]

        # Reminder that `self.visible_layers` is a length two list:
        # [(10, 5, 4), (10, 10, 4)]
        layer_outputs = [self.visible_layers]

        # import pdb; pdb.set_trace()

        for layer in self.layers[:-1]:
            layer_outputs.append(layer.layer_forward(layer_outputs[-1]))

        layer_outputs.append(self.layers[-1].layer_forward(layer_outputs[-1]))

        log.debug(f"Finished propagating through the network")

        # instantiate final matrices:
        log.debug(f"Instantiate the orbitals tensors")
        phi_up = torch.empty(self.num_determinants, self.n_up, self.n_up)
        phi_down = torch.empty(self.num_determinants, self.n_down, self.n_down)

        # compute components of the determinants
        log.debug(f"Compute the orbitals")
        for k in range(self.num_determinants):
            self.compute_up_spin_phis(layer_outputs, phi_up, k)
            self.compute_down_spin_phis(layer_outputs, phi_down, k)

        # compute the determinants:
        log.debug(f"Compute the determinants")
        d_up = torch.det(phi_up)
        d_down = torch.det(phi_down)

        # Weighted sum:
        log.debug(f"Compute the wavefunction")
        wavefunction = torch.sum(self.omega_weights * d_up * d_down)  # sum of result of element-wise multiplications

        log.debug(f"All done!\n{wavefunction = }")
        return wavefunction


class FermiLayer(torch.nn.Module):
    """ x """

    def __init__(self, spin_config, hidden_layer_dimensions):
        """ x """
        super().__init__()

        log.debug("Initializing a FermiLayer\n")

        self.n = sum(spin_config)
        self.n_up, self.n_down = spin_config
        log.debug(f"{hidden_layer_dimensions = }")

        assert self.n > 0, 'must have at least 1 electron'  # just in case

        """ so for each layer it takes in a tuple (20, 4) or (32, 4) as the h in/out dimensions

        The `f_vector_length` will have dimensionality (3*IN, 2*OUT)
        """

        # the hidden layers are specified by input and output layer dimensions
        hidden_input_layers, hidden_output_layers = hidden_layer_dimensions

        # the input layers are specified by two numbers
        # one for the single stream
        # one for the double stream
        in_1_stream_dim, in_2_stream_dim, = hidden_input_layers

        # the out layers are specified by two numbers
        # one for the single stream
        # one for the double stream
        out_1_stream_dim, out_2_stream_dim, = hidden_output_layers

        """ calculate how long the f vector will be
        The f_vector is made up of (h, g1▲, g1▼, g2▲, g2▼) with dimensions
            h   - (batch, 10, 5, 4) ~ (batch, 10, 20)
            g1▲ - (batch, 1, 5, 4)  ~ (batch, 1,  20)
            g1▼ - (batch, 1, 5, 4)  ~ (batch, 1,  20)
            g2▲ - (batch, 0, 10, 4) ~ (batch, 10, 4)
            g2▼ - (batch, 0, 10, 4) ~ (batch, 10, 4)

        Where index 0 means we trace over that index and don't keep it.
        Where we flatten over the 5,4 dimensions to get 20.
        Where we tile the (g1▲, g1▼) over their 1 dimensions (10 copies)


        Note here some confusion that can arise.
        If we have 10 electrons and 5 nuclei then g1▲ and g1▼ are always (batch, 1, 20)
        no matter the distribution of up/down spins among the 10 electrons.
        Since both g1▲ and g1▼ are means over their respective electrons.
        If there are 4 spin up and 6 spin down electrons then:
            g1▲ will be the mean over the first 4 electrons (batch, 0:3, 20)
            g1▼ will be the mean over the last 6 electrons  (batch, 4:9, 20)
        but both will have a size of (batch, 1, 20) after computing the mean.

        The same principle applies for g2▲ and g2▼.
        They will always be (batch, 10, 4) regardless of the spin distribution of the electrons

        So given all that, if we group the components of the f_vector by their leading dimension
            Dim 20:
                2 components (g1▲, g1▼) of dimension (batch, 1,  20)
                1 component  (h)        of dimension (batch, 10, 20)
            Dim 4:
                2 components (g2▲, g2▼) of dimension (batch, 10, 4)

        We have 3 components with dimension (batch, 10, 20)
        We have 2 components with dimension (batch, 10, 4)

        For parameters
            IN:(20, 4) OUT: (32, 4)
        f_vector_length will be
            (3*IN_1 + 2*IN_2) ~ (3*20 + 2*4) ~ (60 + 8) ~ 68
        """
        if (self.n_up == 0) or (self.n_down == 0):
            # case with all electrons having same spin (i.e. no up spins or no down spins)
            f_vector_length = 2*in_1_stream_dim + 1*in_2_stream_dim
        else:
            f_vector_length = 3*in_1_stream_dim + 2*in_2_stream_dim

        # lambda function takes any number of args and passes them on
        p_func = lambda *size: torch.nn.Parameter(torch.rand(size))

        """ matrix and bias vector for each single stream's linear op applied to the f vector
            and yielding a vector of the output length

        self.v_matrices is a vector of matrices
        self.b_vectors is a vector of vectors

        The V matrices will be of dimension
            (10, (3*IN_1 + 2*IN_2), OUT_1)
        The b vectors will be of dimension
            (10, OUT_1)

        so for IN:(20, 4) OUT: (32, 4)

            V - (10, (3*20 + 2*4), 32) ~ (10, (60 + 8), 32) ~ (10, 68, 32)
            b - (10, 32)
        """
        self.v_matrices = p_func(self.n, f_vector_length, out_1_stream_dim)
        self.b_vectors = p_func(self.n, out_1_stream_dim)

        """ matrix and bias vector for each double stream's linear op applied to the f vector
            and yielding a vector of the output length

        self.w_matrices is a matrix of matrices
        self.c_vectors is a matrix of vectors

        The W matrices will be of dimension
            (10, 10, IN_2, OUT_2)
        The c vectors will be of dimension
            (10, 10, OUT_2)

        so for IN:(20, 4) OUT: (32, 4)

            W - (10, 10, 4, 4)
            c - (10, 10, 4)
        """
        self.w_matrices = p_func(self.n, self.n, in_2_stream_dim, out_2_stream_dim)
        self.c_vectors = p_func(self.n, self.n, out_2_stream_dim)

        log.debug(f"{self.v_matrices.shape = }")
        log.debug(f"{self.b_vectors.shape = }")
        log.debug(f"{self.w_matrices.shape = }")
        log.debug(f"{self.c_vectors.shape = }")

    def create_f_vectors(self, single_h, double_h):
        """ Create the f_vector: (h, g1▲, g1▼, g2▲, g2▼)  """
        log.debug(f"input stream shapes:\n")
        log.debug(f"{single_h.shape = }")
        log.debug(f"{double_h.shape = }")

        n_up = self.n_up  # number of up spin electrons
        single_h_up, single_h_down = single_h[:n_up], single_h[n_up:]
        double_h_ups, double_h_downs = double_h[:n_up, :, ...], double_h[n_up:, :, ...]

        log.debug(f"up_down seperation of the streams:\n")
        log.debug(f"{single_h_up.shape = }")
        log.debug(f"{single_h_down.shape = }")
        log.debug(f"{double_h_ups.shape = }")
        log.debug(f"{double_h_downs.shape = }")

        n = len(single_h)  # number of electrons?

        # compute the means (g components)
        summation_dim = 0  # (note that we should change this if we are broadcasting over batch_size)
        single_g_up = torch.mean(single_h_up, summation_dim, keepdim=True) if single_h_up.nelement() else torch.empty(0)
        single_g_down = torch.mean(single_h_down, summation_dim, keepdim=True) if single_h_down.nelement() else torch.empty(0)
        double_g_ups = torch.mean(double_h_ups, summation_dim) if double_h_ups.nelement() else torch.empty(n, 0)
        double_g_downs = torch.mean(double_h_downs, summation_dim) if double_h_downs.nelement() else torch.empty(n, 0)

        log.debug(f"f vector sizes:\n")
        log.debug(f"{single_h.shape = }")
        log.debug(f"{single_g_up.shape = }")
        log.debug(f"{single_g_down.shape = }")
        log.debug(f"{double_g_ups.shape = }")
        log.debug(f"{double_g_downs.shape = }")
        log.debug(f"\n")

        """ creating the tiling dimensions
        This should create a tuple
            (10, 1)
        given `self.n` = 10
        """
        dims = (self.n, 1)
        # dims = (self.n, 1)
        dims = (n, 1)

        """ Since `single_h` will have dimensionality (10, 20)
        but `single_g_down` had dimensionality (1, 20)
        we have to repeat its values
        """
        f_vectors = torch.cat((
            single_h,
            torch.tile(single_g_up, dims),
            torch.tile(single_g_down, dims),
            double_g_ups,
            double_g_downs
        ), axis=1).type(torch.FloatTensor)

        return f_vectors

    def single_stream_output(self, f_vectors, single_h):
        """ x """

        # The V matrices will be of dimension
        #     (10, (3*IN_1 + 2*IN_2), OUT_1)
        # The b vectors will be of dimension
        #     (10, OUT_1)

        # so for IN:(20, 4) OUT: (32, 4)

        #     V - (10, (3*20 + 2*4), 32) ~ (10, (60 + 8), 32) ~ (10, 68, 32)
        #     b - (10, 32)

        log.debug(f"{f_vectors.shape = }")
        log.debug(f"{f_vectors.unsqueeze(-1).shape = }")
        log.debug(f"{self.v_matrices.shape = }")
        log.debug(f"{torch.transpose(self.v_matrices, 1, 2).shape = }")

        """ f_vectors is size (10, 68) for methane
        68 is:
            - 5*4 = 20 for single_h
            - 5*4 = 20 for single_g_up
            - 5*4 = 20 for single_g_down
            -        4 for double_g_ups (because it was a mean)
            -        4 for double_g_downs (because it was a mean)

        indexing like this:
            f_vectors.unsqueeze(-1)
        gives dimensionality
            (10, 68, 1)
        """

        # self.v_matrices is size (10, 68, 32) for methane
        # torch.transpose(self.v_matrices, 1, 2) is size (10, 32, 68) for methane

        """ The matmul will be between
        f_vectors.unsqueeze(-1)
            (10, 68, 1)
        self.v_matrices
            (10, 32, 68)
        with output
            (10, 32, 1)
        """

        # single_output = torch.tanh(torch.bmm(torch.transpose(self.v_matrices, 1, 2), f_vectors[:,:,None]) + self.b_vectors)  # Note: check dimensions order for torch.mul are correct??

        single_stream_output = torch.tanh(
            torch.squeeze(
                # Note: check dimensions order for torch.mul are correct??
                torch.matmul(
                    torch.transpose(self.v_matrices, 1, 2),
                    f_vectors.unsqueeze(-1)
                ),
                dim=2
            ) + self.b_vectors
        )

        # W - (10, 10, 4, 4)
        # c - (10, 10, 4)

        # output[0] = np.tanh(torch.tensor([(self.v_matrices[i] @ f_vectors[i]) + self.b_vectors[i] for i in range(len(f_vectors))]))
        if single_stream_output.size() == single_h.size():
            # single_stream_output += single_h
            single_stream_output = single_stream_output + single_h

        return single_stream_output

    def double_stream_output(self, f_vectors, double_h):
        """ x """

        # double_output = torch.tanh(torch.mul(self.w_matrices, double_h) + self.c_vectors)#Note: check dimensions order for @ (np.matmul) are correct??
        # Note: check dimensions order for torch.mul are correct??
        if self.w_matrices.size()[-1] == 0:
            w_mats_size = self.w_matrices.size()
            double_stream_output = torch.zeros(w_mats_size[0], w_mats_size[1], w_mats_size[-2])
            log.debug(f"{double_stream_output.shape = }")

        else:
            log.debug(f"{self.w_matrices.shape = }")
            log.debug(f"{torch.flatten(self.w_matrices, end_dim=1).shape = }")
            log.debug(f"{double_h.shape = }")
            log.debug(f"{double_h.unsqueeze(-1).shape = }")
            log.debug(f"{torch.flatten(double_h.unsqueeze(-1), end_dim=1).shape = }")
            log.debug(f"{torch.flatten(self.c_vectors, end_dim=1).shape = }")

            double_stream_output = torch.tanh(
                torch.squeeze(
                    torch.matmul(
                        self.w_matrices,
                        double_h.unsqueeze(-1)
                    ),
                    dim=3
                ) + self.c_vectors
            )

        # output[1] = np.tanh(torch.tensor([[(self.w_matrices[i][j] @ double_h[i][j]) + self.c_vectors[i][j] for j in range(len(double_h[0]))] for i in range(len(double_h))]))

        if double_stream_output.size() == double_h.size():
            # double_stream_output += double_h
            double_stream_output = double_stream_output + double_h

        return double_stream_output

    def layer_forward(self, input_tensor):
        """ Propagate a layer """

        # flatten single_h from (batch_size, 10, 5, 4) -> (batch_size, 10, 20)
        single_h = input_tensor[0].type(torch.FloatTensor).flatten(start_dim=1)

        # double_h is (batch_size, 10, 10, 4)
        double_h = input_tensor[1].type(torch.FloatTensor)

        # build f: (h, g1▲, g1▼, g2▲, g2▼)
        f_vectors = self.create_f_vectors(single_h, double_h)

        # build the two streams
        single_stream_output = self.single_stream_output(f_vectors, single_h)
        double_stream_output = self.double_stream_output(f_vectors, double_h)

        return [single_stream_output, double_stream_output]
