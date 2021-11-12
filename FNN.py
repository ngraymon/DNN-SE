import torch
import numpy as np

class FermiNet(torch.nn.Module):
    def __init__(self, L, n_up, electron_positions, nuclei_positions, custom_h_sizes=False, num_determinants=10):
        super(FermiNet, self).__init__()

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
        eN_vectors = np.array([[i-j for j in nuclei_positions] for i in electron_positions])
        self.eN_vectors = torch.from_numpy(eN_vectors)
        self.inputs[0] = torch.from_numpy(np.array([
            np.concatenate([
                np.concatenate([eN_vectors[i][j] for j in range(I)], axis=None), 
                [np.linalg.norm(eN_vectors[i][j]) for j in range(I)]
            ]) for i in range(n)]))
        #double stream inputs:
        ee_vectors = [[i-j for j in electron_positions] for i in electron_positions]
        self.inputs[1] = torch.from_numpy(np.array([[
            np.concatenate([
                ee_vectors[i][j], 
                [np.linalg.norm(ee_vectors[i][j])]
            ]) for j in range(n)] for i in range(n)]))
        
        #Randomly initialise trainable parameters:
        self.final_weights = torch.nn.Parameter(torch.rand(self.num_determinants, n, custom_h_sizes[-1][0]))#w vectors
        self.final_biases = torch.nn.Parameter(torch.rand(self.num_determinants, n))#g scalars
        self.pi_weights = torch.nn.Parameter(torch.rand(self.num_determinants, n, I))#pi scalars for decaying envelopes
        self.sigma_weights = torch.nn.Parameter(torch.rand(self.num_determinants, n, I))#sigma scalars for decaying envelopes
        self.omega_weights = torch.nn.Parameter(torch.rand(self.num_determinants))#omega scalars for summing determinants
    
    def forward(self):
        layer_outputs = [self.inputs]
        for i in self.layers[:-1]:
            layer_outputs.append(i.forward(layer_outputs[-1], self.n_up))
        layer_outputs.append(i.forward(layer_outputs[-1], self.n_up))

        #Compute final matrices:
        phi_up = torch.empty(self.num_determinants, self.n_up, self.n_up)
        phi_down = torch.empty(self.num_determinants, self.n - self.n_up, self.n - self.n_up)
        for k in range(self.num_determinants):
            #up spin:
            for i in range(self.n_up):
                for j in range(self.n_up):
                    final_dot = torch.dot(self.final_weights[k][i], layer_outputs[-1][0][j]) + self.final_biases[k][i]
                    env_sum = torch.sum(torch.stack([self.pi_weights[k][i][m]*torch.exp(-torch.norm(self.sigma_weights[k][i][m] * self.eN_vectors[j][m])) for m in range(self.I)]))
                    phi_up[k][i][j] = final_dot * env_sum
            #down spin:
            for i in range(self.n_up, self.n):
                for j in range(self.n_up, self.n):
                    final_dot = torch.dot(self.final_weights[k][i], layer_outputs[-1][0][j]) + self.final_biases[k][i]
                    env_sum = torch.sum(torch.stack([self.pi_weights[k][i][m]*torch.exp(-torch.norm(self.sigma_weights[k][i][m] * self.eN_vectors[j][m])) for m in range(self.I)]))
                    phi_down[k][i-self.n_up][j-self.n_up] = final_dot * env_sum
        
        #Compute determinants:
        d_up = torch.det(phi_up)
        d_down = torch.det(phi_down)
        #Weighted sum:
        wavefunction = torch.sum(self.omega_weights * d_up * d_down)#sum of result of element-wise multiplications

        return wavefunction



class FermiLayer(torch.nn.Module):
    def __init__(self, n, h_in_dims, h_out_dims):
        super().__init__()

        f_vector_length = 3*h_in_dims[0] + 2*h_in_dims[1]
        #matrix and bias vector for each single stream's linear op, applied to the f vector and yielding a vector of the output length
        self.v_matrices = torch.nn.Parameter(torch.rand(n, f_vector_length, h_out_dims[0]))#vector of matrices
        self.b_vectors = torch.nn.Parameter(torch.rand(n, h_out_dims[0]))#vector of vectors
        ##matrix and bias vector for each double stream's linear op
        self.w_matrices = torch.nn.Parameter(torch.rand(n, n, h_in_dims[1], h_out_dims[1]))#matrix of matrices
        self.c_vectors = torch.nn.Parameter(torch.rand(n, n, h_out_dims[1]))#matrix of vectors
    
    def forward(self, input, n_up):
        #single layers:
        single_h, double_h = input[0].type(torch.FloatTensor), input[1].type(torch.FloatTensor)
        single_h_up = single_h[:n_up]
        single_h_down = single_h[n_up:]
        double_h_ups = double_h[:, :n_up]
        double_h_downs = double_h[:, n_up:]

        single_g_up = torch.mean(single_h_up, 0)
        single_g_down = torch.mean(single_h_down, 0)
        double_g_ups = torch.mean(double_h_ups, 1)#Note: double check which axis?
        double_g_downs = torch.mean(double_h_downs, 1)#Note: double check which axis?

        n = len(input[0])
        f_vectors = torch.stack([
            torch.cat((
                single_h[i], 
                single_g_up, 
                single_g_down, 
                double_g_ups[i], 
                double_g_downs[i]
            )) for i in range(n)]).type(torch.FloatTensor)
        # print(f_vectors.size())
        
        print(torch.transpose(self.v_matrices, 1, 2).size(), f_vectors[:,:,None].size())
        print(self.v_matrices.type(), f_vectors.type())
        # single_output = torch.tanh(torch.bmm(torch.transpose(self.v_matrices, 1, 2), f_vectors[:,:,None]) + self.b_vectors)#Note: check dimensions order for torch.mul are correct??
        single_output = torch.tanh(torch.squeeze(torch.matmul(torch.transpose(self.v_matrices, 1, 2), f_vectors[:,:,None]), 2) + self.b_vectors)#Note: check dimensions order for torch.mul are correct??
        print(single_output.size())
        # output[0] = np.tanh(torch.tensor([(self.v_matrices[i] @ f_vectors[i]) + self.b_vectors[i] for i in range(len(f_vectors))]))
        if single_output.size() == single_h.size():
            single_output += single_h

        #double layers:
        # double_output = torch.tanh(torch.mul(self.w_matrices, double_h) + self.c_vectors)#Note: check dimensions order for @ (np.matmul) are correct??
        print(torch.flatten(self.w_matrices, end_dim=1).size(), torch.flatten(double_h[:,:,:,None], end_dim=1).size())
        print(self.c_vectors.size())
        double_output = torch.tanh(torch.squeeze(torch.bmm(torch.flatten(self.w_matrices, end_dim=1), torch.flatten(double_h[:,:,:,None], end_dim=1)), 2) + torch.flatten(self.c_vectors, end_dim=1))#Note: check dimensions order for torch.mul are correct??
        #reshape:
        shape = list(double_output.shape)[1:]
        double_output = double_output.reshape([n,n]+shape)
        print("a", double_output.size())
        # output[1] = np.tanh(torch.tensor([[(self.w_matrices[i][j] @ double_h[i][j]) + self.c_vectors[i][j] for j in range(len(double_h[0]))] for i in range(len(double_h))]))
        if double_output.size() == double_h.size():
            double_output += double_h

        return [single_output, double_output]

