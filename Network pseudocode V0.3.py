import numpy as np

class FermiNet:
    #Note: the first n_up electrons in electron_positions are up spin, the rest are down spin
    #example: custom_h_sizes = [[4*I, 4], [10, 10], [6, 7], [8, 8], [5, 0]]     (L=4)
    def __init__(self, L, n_up, electron_positions, nuclei_positions, custom_h_sizes=False, num_determinants=10):
        n, I = len(electron_positions), len(nuclei_positions)
        self.n, self.I = n, I
        self.n_up = n_up
        self.num_determinants = num_determinants
        if custom_h_sizes == False:#default configuration, ensures correct dimensions in first layer, and keeps all h vectors the same length
            custom_h_sizes = [[4*I, 4] for i in range(L+1)]
            custom_h_sizes[-1][1] = 0   #final double stream layer not necessary, so setting to zero gives empty weight matrices and saves computations
        self.layers = [FermiLayer(n, custom_h_sizes[i], custom_h_sizes[i+1]) for i in range(L)]

        #inputs in format [single_h_vecs_vector, double_h_vecs_matrix]
        self.inputs = [None, None]#to be processed
        #single stream inputs:
        eN_vectors = [[i-j for j in nuclei_positions] for i in electron_positions]
        self.eN_vectors = eN_vectors
        self.inputs[0] = np.array([
            np.concatenate([
                np.concatenate([eN_vectors[i][j] for j in range(I)], axis=None), 
                [eN_vectors[i][j].linalg.norm() for j in range(I)]
            ]) for i in range(n)])
        #double stream inputs:
        ee_vectors = [[i-j for j in electron_positions] for i in electron_positions]
        self.inputs[1] = np.array([[
            np.concatenate([
                ee_vectors[i][j], 
                [ee_vectors[i][j].linalg.norm()]
            ]) for j in range(n)] for i in range(n)])
        
        #Randomly initialise trainable parameters:
        self.final_weights = np.random.rand(self.num_determinants, n, custom_h_sizes[-1][0])#w vectors
        self.final_biases = np.random.rand(self.num_determinants, n)#g scalars
        self.pi_weights = np.random.rand(self.num_determinants, n, I)#pi scalars for decaying envelopes
        self.sigma_weights = np.random.rand(self.num_determinants, n, I)#sigma scalars for decaying envelopes
        self.omega_weights = np.random.rand(self.num_determinants)#omega scalars for summing determinants
    
    def forward(self):
        layer_outputs = [self.inputs]
        for i in self.layers[:-1]:
            layer_outputs.append(i.forward(layer_outputs[-1], self.n_up))
        layer_outputs.append(i.forward(layer_outputs[-1], self.n_up, double_streams=False))

        #Compute final matrices:
        phi_up, phi_down = [], []
        for k in range(self.num_determinants):
            #up spin:
            phi_up.append([])
            for i in range(self.n_up):
                phi_up[-1].append([])
                for j in range(self.n_up):
                    final_dot = np.dot(self.final_weights[k][i], layer_outputs[-1][0][j]) + self.final_biases[k][i]
                    env_sum = sum([self.pi_weights[k][i][m]*np.exp(-np.linalg.norm(self.sigma_weights[k][i][m] * self.eN_vectors[j][m])) for m in range(self.I)])
                    phi_ij = final_dot * env_sum
                    phi_up[-1][-1].append(phi_ij)
            #down spin:
            phi_down.append([])
            for i in range(self.n_up, self.n):
                phi_down[-1].append([])
                for j in range(self.n_up, self.n):
                    final_dot = np.dot(self.final_weights[k][i], layer_outputs[-1][0][j]) + self.final_biases[k][i]
                    env_sum = sum([self.pi_weights[k][i][m]*np.exp(-np.linalg.norm(self.sigma_weights[k][i][m] * self.eN_vectors[j][m])) for m in range(self.I)])
                    phi_ij = final_dot * env_sum
                    phi_down[-1][-1].append(phi_ij)
        
        #Compute determinants:
        d_up = np.linalg.det(phi_up)
        d_down = np.linalg.det(phi_down)
        #Weighted sum:
        wavefunction = [self.omega_weights[k] * d_up[k] * d_down[k] for k in range(self.num_determinants)]

        return wavefunction

class FermiLayer:
    def __init__(self, n, h_in_dims, h_out_dims):
        f_vector_length = 3*h_in_dims[0] + 2*h_in_dims[1]
        #matrix and bias vector for each single stream's linear op, applied to the f vector and yielding a vector of the output length
        self.v_matrices = np.random.rand(n, f_vector_length, h_out_dims[0])#vector of matrices
        self.b_vectors = np.random.rand(n, h_out_dims[0])#vector of vectors
        ##matrix and bias vector for each double stream's linear op
        self.w_matrices = np.random.rand(n, n, h_in_dims[1], h_out_dims[1])#matrix of matrices
        self.c_vectors = np.random.rand(n, n, h_out_dims[1])#matrix of vectors
    
    def forward(self, input, n_up, double_streams=True):
        output = [None, None]#to be computed

        #single layers:
        single_h, double_h = input[0], input[1]
        single_h_up = single_h[:n_up]
        single_h_down = single_h[n_up:]
        double_h_ups = double_h[:, :n_up]
        double_h_downs = double_h[:, n_up:]

        single_g_up = np.mean(single_h_up, axis=0)
        single_g_down = np.mean(single_h_down, axis=0)
        double_g_ups = np.mean(double_h_ups, axis=1)#Note: double check which axis?
        double_g_downs = np.mean(double_h_downs, axis=1)#Note: double check which axis?

        n = len(input[0])
        f_vectors = np.array([
            np.concatenate([
                single_h[i], 
                single_g_up, 
                single_g_down, 
                double_g_ups[i], 
                double_g_downs[i]
            ]) for i in range(n)])
        
        # output[0] = np.tanh((self.v_matrices @ f_vectors) + self.b_vectors)#Note: check dimensions order for @ (np.matmul) are correct??
        output[0] = np.tanh(np.array([(self.v_matrices[i] @ f_vectors[i]) + self.b_vectors[i] for i in range(len(f_vectors))]))
        if output[0].shape == single_h.shape:
            output[0] += single_h

        #double layers:
        # output[1] = np.tanh((self.w_matrices @ double_h) + self.c_vectors)#Note: check dimensions order for @ (np.matmul) are correct??
        output[1] = np.tanh(np.array([[(self.w_matrices[i][j] @ double_h[i][j]) + self.c_vectors[i][j] for j in range(len(double_h[0]))] for i in range(len(double_h))]))
        if output[1].shape == double_h.shape:
            output[1] += double_h

        return output

