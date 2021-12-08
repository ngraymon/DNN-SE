""" Dead simple tests for train.py """

# system imports
from types import SimpleNamespace

# third party imports
import pytest
from pytest import raises
import torch

# local imports
from . import context
import train


@pytest.fixture
def train_args():
    """ create training arguments """

    network = SimpleNamespace()
    network.parameters = lambda: [torch.zeros(1), ]
    network.zero_grad = lambda: None

    monte_carlo = SimpleNamespace()
    H_operators = [SimpleNamespace(), SimpleNamespace()]

    fake_param = {'lr': 0.3, 'epoch': 0}

    args = [network, monte_carlo, H_operators, fake_param]
    return args


class Test_Train():

    def test__init__(self, train_args):
        """ x """
        train_object = train.Train(*train_args, clip_el=None)

        # assert all arguments got stored correctly
        assert train_object.net is train_args[0]
        assert train_object.mcmc is train_args[1]
        assert train_object.kinetic is train_args[2][0]
        assert train_object.potential is train_args[2][1]
        assert isinstance(train_object.optimizer, torch.optim.Adam)
        assert train_object.clip_el is None

    def test_train_KFAC(self, train_args):
        """ x """

        # create the object
        train_object = train.Train(*train_args, clip_el=None)

        # check return value
        assert 0 == train_object.train_KFAC()

    def test_train_zero_epoch(self, train_args):
        """ x """

        # add epoch parameter to dict
        train_args[3]['epoch'] = 0

        # create the object
        train_object = train.Train(*train_args, clip_el=None)

        loss_array = train_object.train(bool_KFAC=False, clipping=False)

        # If there are no epochs we should return an empty list
        assert [] == loss_array

    def test_train_nan_walkers(self, train_args):
        """ Make sure assert fails if walkers have nan """
        network, monte_carlo, H_operators, param = train_args

        # add epoch parameter to dict
        param['epoch'] = 1

        # modify preform_one_step
        monte_carlo.preform_one_step = lambda: (
            0.0, torch.tensor([float('nan')]), None
        )

        # create the object
        train_object = train.Train(*train_args, clip_el=None)

        # cause the error
        with raises(AssertionError) as e_info:
            train_object.train()

        assert 'state configuration is borked' in str(e_info.value)

    def test_train_two_epoch(self, train_args, monkeypatch):
        """ We will test two loops of the training

        To make this manageable we will monkey patch
        all of the expensive functions.
        """

        return  # not done yet

        network, monte_carlo, H_operators, param = train_args

        # add epoch parameter to dict
        param['epoch'] = 2

        # modify preform_one_step
        monte_carlo.preform_one_step = lambda: [
            torch.zeros(1), torch.zeros(1), torch.zeros(1),
        ]

        # add functions to H_operators
        kinetic, potential = H_operators
        monkeypatch.setattr(monte_carlo, 'preform_one_step', lambda: (0, 0, 0))

        # create the object
        train_object = train.Train(*train_args, clip_el=None)

        loss_array = train_object.train(bool_KFAC=False, clipping=False)

        # If there are no epochs we should return an empty list
        assert [] == loss_array
