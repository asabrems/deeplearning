import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('autograder')
from helpers import *

sys.path.append('./')
from mytorch.nn.conv import *
from mytorch.nn.sequential import Sequential


from mytorch.tensor import Tensor

############################################################################################
##################################   Section 3.1 - Conv1D   ################################
############################################################################################

def conv1d_forward_correctness(num_layers=1):
    '''
    CNN: scanning with a MLP with stride
    '''
    scores_dict = [0]

    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5, 15)
    channels = [np.random.randint(5, 15) for i in range(num_layers + 1)]
    kernel = [np.random.randint(3,7) for i in range(num_layers)]
    stride = [np.random.randint(3,5) for i in range(num_layers)]
    width = np.random.randint(60,80)
    batch_size = np.random.randint(1,4)

    x = np.random.randn(batch_size, channels[0], width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    test_layers = [Conv1d(channels[i], channels[i + 1], kernel[i], stride[i])
                    for i in range(num_layers)]
    test_model = Sequential(*test_layers)


    torch_layers = [nn.Conv1d(channels[i], channels[i + 1], kernel[i], stride=stride[i])
                    for i in range(num_layers)]
    torch_model = nn.Sequential(*torch_layers)

    for torch_layer, test_layer in zip(torch_model, test_model.layers):
        torch_layer.weight = nn.Parameter(torch.tensor(test_layer.weight.data))
        torch_layer.bias = nn.Parameter(torch.tensor(test_layer.bias.data))
    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    y2 = test_model(Tensor(x))
    test_y = y2.data

    # check that model is correctly configured
    check_model_param_settings(test_model)

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1

    return scores_dict

def test_conv1d_forward():
    np.random.seed(11785)
    n = 3
    for i in range(n):
        a = conv1d_forward_correctness(i+1)[0]
        if a != 1:
            print('Failed Conv1D Forward Test: %d / %d' % (i + 1, n))
            return False
        else:
            print('Passed Conv1D Forward Test: %d / %d' % (i + 1, n))
    return True

def conv1d_backward_correctness(num_layers = 1):
    '''
    CNN: scanning with a MLP with stride
    '''
    scores_dict = [0, 0, 0, 0]

    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5, 15)
    channels = [np.random.randint(5, 15) for i in range(num_layers + 1)]
    kernel = [np.random.randint(3,7) for i in range(num_layers)]
    stride = [np.random.randint(3,5) for i in range(num_layers)]
    width = np.random.randint(60,80)
    batch_size = np.random.randint(1,4)

    x = np.random.randn(batch_size, channels[0], width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    test_layers = [Conv1d(channels[i], channels[i + 1], kernel[i], stride[i])
                    for i in range(num_layers)]
    test_model = Sequential(*test_layers)


    torch_layers = [nn.Conv1d(channels[i], channels[i + 1], kernel[i], stride=stride[i])
                    for i in range(num_layers)]
    torch_model = nn.Sequential(*torch_layers)

    for torch_layer, test_layer in zip(torch_model, test_model.layers):
        torch_layer.weight = nn.Parameter(torch.tensor(test_layer.weight.data))
        torch_layer.bias = nn.Parameter(torch.tensor(test_layer.bias.data))

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    b, c, w = torch_y.shape
    y1.sum().backward()
    dx1 = x1.grad
    torch_dx = dx1.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    x_tensor = Tensor(x, requires_grad=True)
    y2 = test_model(x_tensor)
    test_y = y2.data

    # check that model is correctly configured
    check_model_param_settings(test_model)

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1

    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    y2.backward()
    test_dx = x_tensor.grad.data

    # check that model is correctly configured
    check_model_param_settings(test_model)

    if not assertions(test_dx, torch_dx, 'type', 'dx'): return scores_dict
    if not assertions(test_dx, torch_dx, 'shape', 'dx'): return scores_dict
    if not assertions(test_dx, torch_dx, 'closeness', 'dx'): return scores_dict
    scores_dict[1] = 1


    for torch_layer, test_layer in zip(torch_model, test_model.layers):
        torch_dW = torch_layer.weight.grad.detach().numpy()
        torch_db = torch_layer.bias.grad.detach().numpy()
        test_dW = test_layer.weight.grad.data
        test_db = test_layer.bias.grad.data

        if not assertions(test_dW, torch_dW, 'type', 'dW'): return scores_dict
        if not assertions(test_dW, torch_dW, 'shape', 'dW'): return scores_dict
        if not assertions(test_dW, torch_dW, 'closeness', 'dW'): return scores_dict

        if not assertions(test_db, torch_db, 'type', 'db'): return scores_dict
        if not assertions(test_db, torch_db, 'shape', 'db'): return scores_dict
        if not assertions(test_db, torch_db, 'closeness', 'db'): return scores_dict

    scores_dict[2] = 1
    scores_dict[3] = 1

    check_model_param_settings(test_model)

    #############################################################################################
    ##############################    Compare Results   #########################################
    #############################################################################################

    return scores_dict

def test_conv1d_backward():
    np.random.seed(11785)
    n = 3
    for i in range(n):
        a, b, c, d = conv1d_backward_correctness(num_layers=i+1)
        if a != 1:
            print('Failed Conv1D Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1 or c != 1 or d != 1:
            print('Failed Conv1D Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            print('Passed Conv1D Backward Test: %d / %d' % (i + 1, n))
    return True


############################################################################################
##################################   Section  - Flatten   ################################
############################################################################################

def flatten_correctness():
    '''
    Flatten Layer
    '''
    scores_dict = [0, 0, 0, 0]

    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5,15)
    width = np.random.randint(60,80)
    batch_size = np.random.randint(1,4)

    x1d = np.random.randn(batch_size, in_c, width)
    x2d = np.random.randn(batch_size, in_c, width, width)
    x1d_tensor = Tensor(x1d, requires_grad=True)
    x2d_tensor = Tensor(x2d, requires_grad=True)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    test_model = Flatten()

    torch_model = nn.Flatten()

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x1d), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    y1.sum().backward()
    dx1 = x1.grad
    torch_x = dx1.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    y2 = test_model(x1d_tensor)
    test_y = y2.data

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1

    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    y2.backward()
    dx2 = x1d_tensor.grad.data
    test_x = dx2

    if not assertions(test_x, torch_x, 'type', 'x'): return scores_dict
    if not assertions(test_x, torch_x, 'shape', 'x'): return scores_dict
    if not assertions(test_x, torch_x, 'closeness', 'x'): return scores_dict
    scores_dict[1] = 1

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x2d), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    y1.sum().backward()
    dx1 = x1.grad
    torch_x = dx1.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    y2 = test_model(x2d_tensor)
    test_y = y2.data

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[2] = 1

    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    y2.backward()
    dx2 = x2d_tensor.grad.data
    test_x = dx2

    if not assertions(test_x, torch_x, 'type', 'x'): return scores_dict
    if not assertions(test_x, torch_x, 'shape', 'x'): return scores_dict
    if not assertions(test_x, torch_x, 'closeness', 'x'): return scores_dict
    scores_dict[3] = 1

    return scores_dict

def test_flatten():
    np.random.seed(11785)
    n = 3
    for i in range(n):
        a, b, c, d = flatten_correctness()
        if a != 1:
            print('Failed Flatten Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1:
            print('Failed Flatten Backward Test: %d / %d' % (i + 1, n))
            return False
        elif c != 1:
            print('Failed Flatten Forward Test: %d / %d' % (i + 1, n))
            return False
        elif d != 1:
            print('Failed Flatten Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            print('Passed Flatten Test: %d / %d' % (i + 1, n))
    return True
