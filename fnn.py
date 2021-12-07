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

tab = " "*4  # define tab as 4 spaces


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
        """ Randomly initialize trainable parameters """

        # lambda function takes any number of args and passes them on
        p_func = lambda *size: torch.nn.Parameter(torch.rand(size))

        # w vectors have size (nof_determinants)
        last_layer_dim = self.layer_dimensions[-1]
        single_stream_dim = last_layer_dim[0]
        self.final_weights = p_func(nof_determinants, nof_electrons, single_stream_dim)

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

        """ the broadcasting for this operation is as follows
        (B, E, N, C) = (B, E, 1, C) -  (N, C)
            - C is always 3 (representing the three cartesian co-ordinates)
            - E is the # of electrons
            - N is the number of nuclei
            - B is the batch size

        so for 7 batches of methane with 5 atoms and 10 electrons you get
            (7, 10, 5, 3) = (7, 10, 1, 3) - (5, 3)
        """
        log.debug(f"{electron_positions.shape = }")
        log.debug(f"{self.nuclei_positions.shape = }")
        log.debug(
            f"{torch.unsqueeze(electron_positions, -2).shape}"
            " - "
            f"{self.nuclei_positions.shape}"
        )

        self.eN_vectors = torch.unsqueeze(electron_positions, -2) - self.nuclei_positions
        log.debug(f"{self.eN_vectors.shape = }")

        """ if we have (7, 10, 5, 3) then take the norm along the 3 dimension (cartesian co-ordinates)
        and get a (7, 10, 5, 1) output
        see (https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) for more info
        """
        with torch.no_grad():
            norm_eN_vectors = torch.linalg.vector_norm(self.eN_vectors, dim=-1, keepdim=True)
        log.debug(f"{norm_eN_vectors.shape = }")

        # concatenate (7, 10, 5, 3) with (7, 10, 5, 1) to get (7, 10, 5, 4)
        single_stream = torch.cat((norm_eN_vectors, self.eN_vectors), dim=-1)
        log.debug(f"{single_stream.shape = }")

        if False:
            # if instead you want to end up with (10, 20)
            # where 20 is 5*3 positions concatenated with 5*1 norms
            single_stream = torch.concat((self.eN_vectors.flatten(1), norm_eN_vectors.flatten(1)), dim=1)
            log.debug(f"{single_stream.shape = }")

        return single_stream

    def _preprocess_double_stream(self, electron_positions):
        """ Create the torch tensor storing the distance between pairs of electrons """

        """ the broadcasting for this operation is as follows
        (B, e1, e2, C) = (B, e1, 1, C) - (B, 1, e2, C)
            - C is always 3 (representing the three cartesian co-ordinates)
            - e1 is the # of electrons
            - e2 is the # of electrons
            - B is the batch size

        so for 7 batches of methane with 10 electrons you get
            (7, 10, 10, 3) = (7, 10, 1, 3) - (7, 1, 10, 3)
        """
        log.debug(f"{electron_positions.shape = }")
        log.debug(f"{electron_positions.shape = }")
        log.debug(
            f"{torch.unsqueeze(electron_positions, -2).shape}"
            " - "
            f"{torch.unsqueeze(electron_positions, -3).shape}"
        )

        self.ee_vectors = torch.unsqueeze(electron_positions, -2) - torch.unsqueeze(electron_positions, -3)
        log.debug(f"{self.ee_vectors.shape = }")

        """ if we have (7, 10, 10, 3) then take the norm along the 3 dimension (cartesian co-ordinates)
        and get a (7, 10, 10, 1) output
        see (https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) for more info
        """
        with torch.no_grad():
            norm_ee_vectors = torch.linalg.vector_norm(self.ee_vectors, dim=-1, keepdim=True)
        log.debug(f"{norm_ee_vectors.shape = }")
        # for i in range(7):
        #     log.debug(f"\n{self.ee_vectors[i] = }")
        #     import pdb; pdb.set_trace()

        # assert not torch.isclose(norm_ee_vectors, torch.zeros_like(norm_ee_vectors)).any(), 'oh no were fucked'
        # log.debug(f"{norm_ee_vectors = }")

        # concatenate (7, 10, 10, 3) with (7, 10, 10, 1) to get (7, 10, 10, 4)
        double_stream = torch.cat((norm_ee_vectors, self.ee_vectors), dim=-1)
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

        # log.debug(f"0\n{electron_positions[0]}")
        log.debug(f"{electron_positions.shape}")

        # single_h_vecs_vector
        single_h_stream = self._preprocess_single_stream(electron_positions)

        # we need to flatten the last two dimensions
        single_h_stream = single_h_stream.flatten(start_dim=-2)

        # double_h_vecs_matrix
        double_h_stream = self._preprocess_double_stream(electron_positions)

        # debug
        log.debug(f"{single_h_stream.shape = }")
        log.debug(f"{double_h_stream.shape = }")

        """ Reminder that the two streams have different dimensionality
        and therefore we cannot simply concatenate them together.
        Using our prior examples the sizes are:
            single_h_stream: (7, 10, 5*4)
            double_h_stream: (7, 10, 10, 4)

        Given
        - 4 is 3 cartesian co-ordinates and its respective 1-norm
        - 10 is the # of electrons
        - 5 is the number of nuclei
        - 7 is the batch size

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

    def compute_orbitals(self, layer_outputs, phi_up, phi_down):
        """ up spin """

        # determinant index
        weights = self.final_weights
        biases = self.final_biases

        pi_weights = self.pi_weights
        sigma_weights = self.sigma_weights

        # log.debug(f"For determinant {k+1}:")
        log.debug(f"{self.final_weights.shape = } {self.final_biases.shape = }")
        log.debug(f"{weights.shape = } {biases.shape = }")
        log.debug(f"{self.pi_weights.shape = } {self.sigma_weights.shape = }")
        log.debug(f"{pi_weights.shape = } {sigma_weights.shape = }")

        if False:  # for debugging
            for i, layer in enumerate(layer_outputs):
                log.debug(f"layer {i} single stream shape: {layer[0].shape}")
                log.debug(f"layer {i} double stream shape: {layer[1].shape}")

        # has dimension of size [(7, 10, 32), (7, 10, 10, 4, 0)]
        last_layer = layer_outputs[-1]

        # has dimension of (7, 10, 32)
        last_layers_single_stream = last_layer[0]
        log.debug(f"{last_layers_single_stream.shape = }")

        # broadcasted dot product; the (w * h + g) term in equation 6 in the paper
        # has an output shape of
        # (batch_size, num_determinants, n_electrons, n_electrons)
        # (7, 16, 10, 10)
        weighted_network_output = biases[None, :, :, None] + torch.mul(
            # has dimension of (batch_size, 1, 1, 10, 32)
            last_layers_single_stream[:, None, None, :, :],
            # has dimension of (1, 16, 10, 1, 32)
            weights[None, :, :, None, :]
        ).sum(-1)
        log.debug(f"{weighted_network_output.shape = }")

        # broadcasted operations; the "sum over m" term in equation 6 in the paper
        decaying_potentials = torch.mul(
            # has dimension of (1, 16, 10, 1, 5)
            pi_weights[None, :, :, None, :],
            torch.exp(
                -torch.norm(torch.mul(
                        # has dimension of (1, 16, 10, 1, 5, 1)
                        sigma_weights[None, :, :, None, :, None],
                        # has dimension of (7, 1, 1, 10, 5, 3)
                        self.eN_vectors[:, None, None, :, :, :]
                    ),
                    dim=-1  # take the norm over the (x,y,z) dimension
                )
            )
        ).sum(-1)  # preform the sum over the nuclei dimension
        log.debug(f"{decaying_potentials.shape = }")

        orbitals = torch.mul(weighted_network_output, decaying_potentials)
        log.debug(f"{orbitals.shape = }")

        phi_up[:] = orbitals[:, :, :self.n_up, :self.n_up]
        phi_down[:] = orbitals[:, :, self.n_up:, self.n_up:]

    def forward(self, electron_positions):
        """ x """

        self.batch_size = electron_positions.shape[0]

        if electron_positions is not None:
            self.preprocess(electron_positions)

        # Reminder that `self.visible_layers` is a length two list:
        # [(7, 10, 5*4), (7, 10, 10, 4)] for methane with 7 batches
        layer_outputs = [self.visible_layers]

        for layer in self.layers[:-1]:
            layer_outputs.append(layer.layer_forward(layer_outputs[-1]))

        layer_outputs.append(self.layers[-1].layer_forward(layer_outputs[-1]))

        log.debug(f"Finished propagating through the network\n")

        # instantiate final matrices:
        log.debug(f"Instantiate the orbitals tensors")
        phi_up = torch.empty(self.batch_size, self.num_determinants, self.n_up, self.n_up)
        phi_down = torch.empty(self.batch_size, self.num_determinants, self.n_down, self.n_down)
        log.debug(f"{phi_up.shape = }")
        log.debug(f"{phi_down.shape = }")
        log.debug("\n")

        # compute components of the determinants
        log.debug(f"Compute the orbitals")
        self.compute_orbitals(layer_outputs, phi_up, phi_down)
        log.debug("\n")

        # compute the determinants:
        log.debug(f"Compute the determinants")
        d_up = torch.det(phi_up)
        d_down = torch.det(phi_down)
        assert (d_up != 0.0).all(), 'singular up matrices'
        assert (d_down != 0.0).all(), 'singular up matrices'
        log.debug(f"{d_up.shape = }")
        log.debug(f"{d_down.shape = }")
        log.debug("\n")

        det = d_up*d_down

        log_det = torch.zeros(self.batch_size)
        abs_det = torch.abs(det)

        for i in range(self.batch_size):
            # no_max_mask = torch.arange(num_replicas)
            # det_no_max = det[i, det[i] != max_val]
            # print(max_val)
            # print(det_no_max.shape)
            abs_max_det = max(abs_det[i])
            log_det[i] = torch.log(abs_max_det) + torch.log(torch.abs(torch.sum(
                self.omega_weights
                * torch.sign(det[i])
                * torch.exp(
                    torch.log(abs_det[i]) - torch.log(abs_max_det)
                )
            )))

        wavefunction = log_det

        # Weighted sum:
        log.debug(f"Compute the wavefunction")

        # double check they all have gradients?
        assert d_up.requires_grad is True
        assert d_down.requires_grad is True
        assert self.omega_weights.requires_grad is True

        # wavefunction = torch.sum(self.omega_weights * d_up * d_down, dim=-1)  # sum of result of element-wise multiplications
        # normed_wavefunction = torch.sum(
        #     torch.nn.functional.normalize(self.omega_weights * d_up * d_down),
        #     dim=-1
        # )

        log.debug(f"{wavefunction.requires_grad = }")
        log.debug(f"{wavefunction.shape = }")

        log.debug(f"All done!\n{wavefunction = }")
        # log.debug(f"All done!\n{normed_wavefunction = }")
        # import pdb; pdb.set_trace()
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

        So f_vector final dimension would be (batch, 10, 68)
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
            (batch, 10, (3*IN_1 + 2*IN_2), OUT_1)
        The b vectors will be of dimension
            (batch, 10, OUT_1)

        so for IN:(20, 4) OUT: (32, 4)

            V - (batch, 10, (3*20 + 2*4), 32)
              ~ (batch, 10, (60 + 8), 32)
              ~ (batch, 10, 68, 32)

            b - (batch, 10, 32)
        """
        self.v_matrices = p_func(1, self.n, f_vector_length, out_1_stream_dim)
        self.b_vectors = p_func(1, self.n, out_1_stream_dim)

        """ matrix and bias vector for each double stream's linear op applied to the f vector
            and yielding a vector of the output length

        self.w_matrices is a matrix of matrices
        self.c_vectors is a matrix of vectors

        The W matrices will be of dimension
            (batch, 10, 10, IN_2, OUT_2)
        The c vectors will be of dimension
            (batch, 10, 10, OUT_2)

        so for IN:(20, 4) OUT: (32, 4)

            W - (batch, 10, 10, 4, 4)
            c - (batch, 10, 10, 4)
        """
        self.w_matrices = p_func(1, self.n, self.n, in_2_stream_dim, out_2_stream_dim)
        self.c_vectors = p_func(1, self.n, self.n, out_2_stream_dim)

        log.debug(f"{self.v_matrices.shape = }")
        log.debug(f"{self.b_vectors.shape = }")
        log.debug(f"{self.w_matrices.shape = }")
        log.debug(f"{self.c_vectors.shape = }")

    def create_f_vectors(self, single_h, double_h):
        """ Create the f_vector: (h, g1▲, g1▼, g2▲, g2▼)  """
        log.debug("\n")
        log.debug(f"input stream shapes:")
        log.debug(f"{single_h.shape = }")
        log.debug(f"{double_h.shape = }")

        n_up = self.n_up  # number of up spin electrons
        single_h_up, single_h_down = single_h[:, :n_up, ...], single_h[:, n_up:, ...]
        double_h_ups, double_h_downs = double_h[:, :n_up, ...], double_h[:, n_up:, ...]

        log.debug("\n")
        log.debug(f"up_down seperation of the streams:")
        log.debug(f"{single_h_up.shape = }")
        log.debug(f"{single_h_down.shape = }")
        log.debug(f"{double_h_ups.shape = }")
        log.debug(f"{double_h_downs.shape = }")

        # compute the means (g components)
        summation_dim = 1  # (note that we should change this if we are broadcasting over batch_size)
        single_g_up = torch.mean(single_h_up, summation_dim, keepdim=True) if single_h_up.nelement() else torch.empty(0)
        single_g_down = torch.mean(single_h_down, summation_dim, keepdim=True) if single_h_down.nelement() else torch.empty(0)
        double_g_ups = torch.mean(double_h_ups, summation_dim) if double_h_ups.nelement() else torch.empty(len(single_h), 0)
        double_g_downs = torch.mean(double_h_downs, summation_dim) if double_h_downs.nelement() else torch.empty(len(single_h), 0)

        dims = (1, self.n, 1)  # This should create a tuple (1, 10, 1) given `self.n` = 10

        log.debug("\n")
        log.debug(f"f vector sizes:")
        log.debug(f"{single_h.shape = }")
        log.debug(f"{torch.tile(single_g_up, dims).shape = }")
        log.debug(f"{torch.tile(single_g_down, dims).shape = }")
        log.debug(f"{double_g_ups.shape = }")
        log.debug(f"{double_g_downs.shape = }")
        log.debug(f"\n")

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
        ), axis=-1).type(torch.FloatTensor)

        return f_vectors

    def single_stream_output(self, f_vectors, single_h):
        """ x """

        # The V matrices will be of dimension
        #     (batch, 10, (3*IN_1 + 2*IN_2), OUT_1)
        # The b vectors will be of dimension
        #     (batch, 10, OUT_1)

        # so for IN:(20, 4) OUT: (32, 4)

        #     V - (batch, 10, (3*20 + 2*4), 32)
        #       ~ (batch, 10, (60 + 8), 32)
        #       ~ (batch, 10, 68, 32)
        #     b - (batch, 10, 32)

        log.debug(f"{f_vectors.shape = }")
        log.debug(f"{f_vectors.unsqueeze(-1).shape = }")
        log.debug(f"{self.v_matrices.shape = }")
        log.debug(f"{torch.transpose(self.v_matrices, -2, -1).shape = }")

        """ f_vectors is size (batch, 10, 68) for methane
        68 is:
            - 5*4 = 20 for single_h
            - 5*4 = 20 for single_g_up
            - 5*4 = 20 for single_g_down
            -        4 for double_g_ups (because it was a mean)
            -        4 for double_g_downs (because it was a mean)

        indexing like this:
            f_vectors.unsqueeze(-1)
        gives dimensionality
            (batch, 10, 68, 1)
        """

        # self.v_matrices is size (batch, 10, 68, 32) for methane
        # torch.transpose(self.v_matrices, -2, -1) is size (batch, 10, 32, 68) for methane

        """ The matmul will be between
        f_vectors.unsqueeze(-1)
            (batch, 10, 68, 1)
        self.v_matrices
            (batch, 10, 32, 68)
        with output
            (batch, 10, 32, 1)
        """

        # single_output = torch.tanh(torch.bmm(torch.transpose(self.v_matrices, 1, 2), f_vectors[:,:,None]) + self.b_vectors)  # Note: check dimensions order for torch.mul are correct??

        single_stream_output = torch.tanh(
            torch.squeeze(
                # Note: check dimensions order for torch.mul are correct??
                torch.matmul(
                    torch.transpose(self.v_matrices, -2, -1),
                    f_vectors.unsqueeze(-1)
                ),
                dim=-1
            ) + self.b_vectors
        )

        # W - (batch, 10, 10, 4, 4)
        # c - (batch, 10, 10, 4)

        # output[0] = np.tanh(torch.tensor([(self.v_matrices[i] @ f_vectors[i]) + self.b_vectors[i] for i in range(len(f_vectors))]))

        if single_stream_output.size() == single_h.size():
            # single_stream_output += single_h
            single_stream_output = single_stream_output + single_h

        return single_stream_output

    def double_stream_output(self, f_vectors, double_h):
        """ x """

        log.debug("\n")  # spacing log for readability

        if self.w_matrices.size()[-1] == 0:
            batch_size = f_vectors.shape[0]
            log.debug(f"{self.w_matrices.shape = }")
            dim = (batch_size, *self.w_matrices.shape[1:])
            double_stream_output = torch.zeros(dim)
            log.debug(f"{double_stream_output.shape = }")

        else:
            log.debug(f"{self.w_matrices.shape = }")
            log.debug(f"{double_h.shape = }")
            log.debug(f"{double_h.unsqueeze(-1).shape = }")
            log.debug(f"{torch.matmul(self.w_matrices, double_h.unsqueeze(-1)).shape = }")
            log.debug(f"{torch.matmul(self.w_matrices, double_h.unsqueeze(-1)).squeeze(-1).shape = }")
            log.debug(f"{self.c_vectors.shape = }")
            # log.debug(f"{torch.flatten(double_h.unsqueeze(-1), start_dim=1, end_dim=2).shape = }")
            # log.debug(f"{torch.flatten(self.c_vectors, start_dim=1, end_dim=2).shape = }")
            # log.debug(f"{self.c_vectors.flatten(start_dim=1, end_dim=2).shape = }")

            double_stream_output = torch.tanh(
                torch.matmul(
                    self.w_matrices,
                    double_h.unsqueeze(-1)
                ).squeeze(-1) + self.c_vectors
            )

        # output[1] = np.tanh(torch.tensor([[(self.w_matrices[i][j] @ double_h[i][j]) + self.c_vectors[i][j] for j in range(len(double_h[0]))] for i in range(len(double_h))]))

        if double_stream_output.size() == double_h.size():
            # double_stream_output += double_h
            double_stream_output = double_stream_output + double_h

        return double_stream_output

    def layer_forward(self, input_tensor):
        """ Propagate a layer """

        # flatten single_h from (batch_size, 10, 5, 4) -> (batch_size, 10, 20)
        single_h = input_tensor[0].type(torch.FloatTensor)
        log.debug(f"{single_h.shape = }")

        # double_h is (batch_size, 10, 10, 4)
        double_h = input_tensor[1].type(torch.FloatTensor)
        log.debug(f"{double_h.shape = }")

        # build f: (h, g1▲, g1▼, g2▲, g2▼)
        f_vectors = self.create_f_vectors(single_h, double_h)

        # build the two streams
        single_stream_output = self.single_stream_output(f_vectors, single_h)
        double_stream_output = self.double_stream_output(f_vectors, double_h)

        log.debug(f"layer_outputs: [{single_stream_output.shape}, {double_stream_output.shape}]\n")
        return [single_stream_output, double_stream_output]
