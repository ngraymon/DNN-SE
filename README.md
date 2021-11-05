# Ab-Initio Solution of the Many-Electron Schr¨odinger Equation with Deep Neural
Networks
here the intersting aspect of this code is that we are using the formula given from the paper for the gradient of the loss.
so what we do is we take the loss function in the sense of pytorch, where the gradient is stored in each weight, relative to the psi (logpsi technically) so psi is our effective loss function, with a factor being added infront of every weight given by the difference between the local energy for that position and the expectation value of the local energy. Then we can simply apply SGD to weights with these losses.

KFAC: this is applied as a preconditioner to the optimizer, meaning we edit the gradients before backpropagating step, this is not yet really clear in my mind of its implementation but I am following: https://tfjgeorge.github.io/assets/docs/EKFAC-NeurIPS2018.pdf, as an inspiration for the KFAC class.
