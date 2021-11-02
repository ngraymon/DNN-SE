""" fermi_nerual_net module

This module contains the network and associated functions.
Import this module as follows:
    import fermi_neural_net as fnn

"""

# system imports

# third party imports
import torch.nn as nn

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
