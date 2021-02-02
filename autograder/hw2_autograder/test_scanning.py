import os
import sys

sys.path.append('./')
sys.path.append('autograder')
sys.path.append('hw2')
sys.path.append('handin')


import numpy as np
import torch

from helpers import *
from mytorch.nn.functional import *
from mytorch.tensor import Tensor
from mytorch.autograd_engine import *

from mlp_scan import CNN_SimpleScanningMLP, CNN_DistributedScanningMLP


def test_simple_scanning_mlp():
    cnn = CNN_SimpleScanningMLP()

    # Load and init weights
    weights = np.load(os.path.join('autograder', 'hw2_autograder', 'weights', 'mlp_weights_part_b.npy'), allow_pickle = True)
    weights = tuple(w.T for w in weights)
    cnn.init_weights(weights)

    # load data and expected answer
    data = np.loadtxt(os.path.join('autograder', 'hw2_autograder', 'data', 'data.asc')).T.reshape(1, 24, -1)
    data = Tensor(data, requires_grad=False, is_parameter=False, is_leaf=True)

    expected_result = np.load(os.path.join('autograder', 'hw2_autograder', 'ref_result', 'res_b.npy'), allow_pickle=True)
    # get forward output and check
    result = cnn(data)

    # check that model is correctly configured
    check_model_param_settings(cnn)

    # now check correct results
    try:
        # check that output is correct
        assert type(result.data) == type(expected_result), f"Incorrect output type: {result.data}, expected: {expected_result}"
        assert result.data.shape == expected_result.shape, f"Incorrect output shape: {result.data.shape}, expected: {expected_result.shape}"
        assert np.allclose(result.data, expected_result), f"Incorrect output values: {result.data}, expected: {expected_result}"
    except Exception as e:
        traceback.print_exc()
        return False

    return True



def test_distributed_scanning_mlp():
    cnn = CNN_DistributedScanningMLP()

    weights = np.load(os.path.join('autograder', 'hw2_autograder', 'weights', 'mlp_weights_part_c.npy'), allow_pickle=True)
    weights = tuple(w.T for w in weights)

    cnn.init_weights(weights)

    data = np.loadtxt(os.path.join('autograder', 'hw2_autograder', 'data', 'data.asc')).T.reshape(1, 24, -1)
    data = Tensor(data, requires_grad=False, is_parameter=False, is_leaf=True)

    expected_result = np.load(os.path.join('autograder', 'hw2_autograder', 'ref_result', 'res_c.npy'), allow_pickle=True)

    result = cnn(data)

    # check that model is correctly configured
    check_model_param_settings(cnn)

    # if passes tests, return true.
    # If exception anywhere (failure or crash), return false
    try:
        # check that output is correct
        assert type(result.data) == type(expected_result), "Incorrect output type."
        assert result.data.shape == expected_result.shape, "Incorrect output shape."
        assert np.allclose(result.data, expected_result), "Incorrect output values."
        #assert np.allclose(result.data, expected_result), f"Incorrect output values: {result.data}, expected: {expected_result}"
    except Exception as e:
        traceback.print_exc()
        return False

    return True