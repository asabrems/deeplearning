import multiprocessing as mtp
import sys

sys.path.append('autograder')
sys.path.append('./')

import traceback
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from helpers import *
from mytorch.nn.activations import *
from mytorch.nn.batchnorm import BatchNorm1d
from mytorch.nn.dropout import Dropout
from mytorch.nn.linear import Linear
from mytorch.nn.loss import *
from mytorch.nn.sequential import Sequential
from mytorch.optim.adam import Adam
from mytorch.tensor import Tensor


# Dropout tests
def test_dropout_forward():
    np.random.seed(11785)
    
    # run on small model forward only
    x = Tensor.randn(5, 10)
    model = Sequential(Linear(10, 5), ReLU(), Dropout(p=0.6))
    my_output = model(x)
    
    # check that model is correctly configured
    check_model_param_settings(model)

    test_output = load_numpy_array('autograder/hw1_bonus_autograder/outputs/dropout_forward.npy')
    return assertions_all(my_output.data, test_output, "test_dropout_forward", 1e-5, 1e-6)

def test_dropout_forward_backward():
    np.random.seed(11785)
    
    # run on small model, forward backward (no step)
    model = Sequential(Linear(10, 20), ReLU(), Dropout(p=0.6))
    x, y = generate_dataset_for_mytorch_model(model, 5)
    x, y = Tensor(x), Tensor(y)
    criterion = CrossEntropyLoss()
    out = model(x)
    
    # check that model is correctly configured
    check_model_param_settings(model)
    
    test_out = load_numpy_array('autograder/hw1_bonus_autograder/outputs/backward_output.npy')
    
    if not assertions_all(out.data, test_out, "test_dropout_forward_backward_output", 1e-5, 1e-6):
        return False
    
    loss = criterion(out, y)
    loss.backward()
    
    # check that model is correctly configured
    check_model_param_settings(model)
    
    assert model[0].weight.grad is not None, "Linear layer must have gradient."
    assert model[0].weight.grad.grad is None, "Final gradient tensor must not have its own gradient"
    assert model[0].weight.grad.grad_fn is None, "Final gradient tensor must not have its own grad function"
    assert model[0].weight.requires_grad, "Weight tensor must have requires_grad==True"
    assert model[0].weight.is_parameter, "Weight tensor must be marked as a parameter tensor"

    test_grad = load_numpy_array('autograder/hw1_bonus_autograder/outputs/backward_grad.npy')
    
    return assertions_all(model[0].weight.grad.data, test_grad, "test_dropout_forward_backward_grad", 1e-5, 1e-6)

def test_big_model_step():
    np.random.seed(11785)
    
    # run a big model
    model = Sequential(Linear(10, 15), ReLU(), Dropout(p=0.2), 
                       Linear(15, 20), ReLU(), Dropout(p=0.1))
    x, y = generate_dataset_for_mytorch_model(model, 4)
    x, y = Tensor(x), Tensor(y)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    
    # check output correct
    out = model(x)
    test_out = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_output.npy')

    # check that model is correctly configured
    check_model_param_settings(model)

    if not assertions_all(out.data, test_out, "test_big_model_step_out", 1e-5, 1e-6):
        return False
    
    # run backward
    loss = criterion(out, y)
    loss.backward()
    
    # check that model is correctly configured
    check_model_param_settings(model)
    
    # check params are correct (sorry this is ugly)
    assert model[0].weight.grad is not None, "Linear layer must have gradient."
    assert model[0].weight.grad.grad is None, "Final gradient tensor must not have its own gradient"
    assert model[0].weight.grad.grad_fn is None, "Final gradient tensor must not have its own grad function"
    assert model[0].weight.requires_grad, "Weight tensor must have requires_grad==True"
    assert model[0].weight.is_parameter, "Weight tensor must be marked as a parameter tensor"
    assert model[3].weight.grad is not None, "Linear layer must have gradient."
    assert model[3].weight.grad.grad is None, "Final gradient tensor must not have its own gradient"
    assert model[3].weight.grad.grad_fn is None, "Final gradient tensor must not have its own grad function"
    assert model[3].weight.requires_grad, "Weight tensor must have requires_grad==True"
    assert model[3].weight.is_parameter, "Weight tensor must be marked as a parameter tensor"
    
    # check gradient for linear layer at idx 0 is correct
    test_grad = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_grad.npy')
    if not assertions_all(model[0].weight.grad.data, test_grad, "test_big_model_grad_0", 1e-5, 1e-6):
        return False
    
    # check gradient for linear layer at idx 3 is correct
    test_grad = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_grad_3.npy')
    if not assertions_all(model[3].weight.grad.data, test_grad, "test_big_model_grad_3", 1e-5, 1e-6):
        return False

    # weight update with adam
    optimizer.step()
    
    # check updated weight values
    assert model[0].weight.requires_grad, "Weight tensor must have requires_grad==True"
    assert model[0].weight.is_parameter, "Weight tensor must be marked as a parameter tensor"

    test_weights_3 = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_weight_update_3.npy')
    test_weights_0 = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_weight_update_0.npy')
    
    return assertions_all(model[0].weight.data, test_weights_0, "test_big_weight_update_0", 1e-5, 1e-6) and \
        assertions_all(model[3].weight.data, test_weights_3, "test_big_weight_update_3", 1e-5, 1e-6)

##############################
# Utilities for testing MLPs #
##############################

def generate_dataset_for_mytorch_model(mytorch_model, batch_size):
    """
    Generates a fake dataset to test on.

    Returns x: ndarray (batch_size, in_features),
            y: ndarray (batch_size,)
    where in_features is the input dim of the mytorch_model, and out_features
    is the output dim.
    """
    in_features = get_mytorch_model_input_features(mytorch_model)
    out_features = get_mytorch_model_output_features(mytorch_model)
    x = np.random.randn(batch_size, in_features)
    y = np.random.randint(out_features, size=(batch_size,))
    return x, y

def get_mytorch_model_input_features(mytorch_model):
    """
    Returns in_features for the first linear layer of a mytorch
    Sequential model.
    """
    return get_mytorch_linear_layers(mytorch_model)[0].in_features


def get_mytorch_model_output_features(mytorch_model):
    """
    Returns out_features for the last linear layer of a mytorch
    Sequential model.
    """
    return get_mytorch_linear_layers(mytorch_model)[-1].out_features


def get_mytorch_linear_layers(mytorch_model):
    """
    Returns a list of linear layers for a mytorch model.
    """
    return list(filter(lambda x: isinstance(x, Linear), mytorch_model.layers))


def get_pytorch_linear_layers(pytorch_model):
    """
    Returns a list of linear layers for a pytorch model.
    """
    return list(filter(lambda x: isinstance(x, nn.Linear), pytorch_model))
