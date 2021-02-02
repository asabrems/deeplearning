#!/usr/bin/env python3

import sys

sys.path.append('autograder')
sys.path.append('hw1')

from helpers import *
from test_mlp import *
from test_autograd import *

################################################################################
#################################### DO NOT EDIT ###############################
################################################################################

tests = [
    # Add operation is a freebie
    {
        'name': '(Freebie) - Addition (0 is full score!)',
        'autolab': 'addtest',
        'handler': test_add,
        'value': 0,
    },
    {
        'name': '1.1.1 - Subtraction',
        'autolab': 'subtest',
        'handler': test_sub,
        'value': 5,
    },
    {
        'name': '1.1.2 - Multiplication',
        'autolab': 'multest',
        'handler': test_mul,
        'value': 5,
    },
    {
        'name': '1.1.3 - Division',
        'autolab': 'divtest',
        'handler': test_div,
        'value': 5,
    },
    {
        'name': '1.2 - Autograd',
        'autolab': 'autogradtest',
        'handler': test_autograd,
        'value': 25,
    },
    {
        'name': '2.1 - Linear Forward',
        'autolab': 'Linear Forward',
        'handler': test_linear_forward,
        'value': 2,
    },
    {
        'name': '2.1 - Linear Backward',
        'autolab': 'Linear Backward',
        'handler': test_linear_backward,
        'value': 8,
    },
    {
        'name': '2.2 - Linear->ReLU Forward',
        'autolab': 'Linear ReLU Forward',
        'handler': test_linear_relu_forward,
        'value': 1,
    },
    {
        'name': '2.2 - Linear->ReLU Backward',
        'autolab': 'Linear ReLU Backward',
        'handler': test_linear_relu_backward,
        'value': 1,
    },
    {
        'name': '2.2 - Linear->ReLU->Linear->ReLU Forward',
        'autolab': 'Big Linear ReLU Forward',
        'handler': test_big_linear_relu_forward,
        'value': 1,
    },
    {
        'name': '2.2 - Linear->ReLU->Linear->ReLU Backward',
        'autolab': 'Big Linear ReLU Backward',
        'handler': test_big_linear_relu_backward,
        'value': 2,
    },
    {
        'name': '2.3 - Linear->ReLU SGD Step',
        'autolab': 'Linear ReLU SGD Step',
        'handler': test_linear_relu_step,
        'value': 5,
    },
    {
        'name': '2.3 - Linear->ReLU->Linear->ReLU SGD Step',
        'autolab': 'Big Linear ReLU SGD Step',
        'handler': test_big_linear_relu_step,
        'value': 5,
    },
    # {
    #     'name': '2.4 - Linear->Batchnorm->ReLU Forward (Train)',
    #     'autolab': 'Linear Batchnorm ReLU Forward (Train)',
    #     'handler': test_linear_batchnorm_relu_forward_train,
    #     'value': 1,
    # },
    # {
    #     'name': '2.4 - Linear->Batchnorm->ReLU Backward (Train)',
    #     'autolab': 'Linear Batchnorm ReLU Backward (Train)',
    #     'handler': test_linear_batchnorm_relu_backward_train,
    #     'value': 1,
    # },
    # {
    #     'name': '2.4 - Linear->Batchnorm->ReLU (Train/Eval)',
    #     'autolab': 'Linear Batchnorm ReLU Train/Eval',
    #     'handler': test_linear_batchnorm_relu_train_eval,
    #     'value': 2,
    # },
    # {
    #     'name': '2.4 - Linear->Batchnorm->ReLU->Linear->Batchnorm->ReLU (Train/Eval)',
    #     'autolab': 'Big Linear Batchnorm ReLU Train/Eval',
    #     'handler': test_big_linear_batchnorm_relu_train_eval,
    #     'value': 6,
    # },
    {
        'name': '2.5 - Linear XELoss Forward',
        'autolab': 'Linear XELoss Forward',
        'handler': test_linear_xeloss_forward,
        'value': 4,
    },
    {
        'name': '2.5 - Linear XELoss Backward',
        'autolab': 'Linear XELoss Backward',
        'handler': test_linear_xeloss_backward,
        'value': 6,
    },
    # {
    #     'name': '2.5 - Linear->Batchnorm->ReLU->Linear->Batchnorm->ReLU XELoss (Train/Eval)',
    #     'autolab': 'Big Linear BN Relu XELoss Train/Eval',
    #     'handler': test_big_linear_bn_relu_xeloss_train_eval,
    #     'value': 5,
    # },
    {
        'name': '2.5 - Linear->ReLU->Linear->ReLU XELoss (Train/Eval)',
        'autolab': 'Big Linear Relu XELoss Train/Eval',
        'handler': test_big_linear_relu_xeloss_train_eval,
        'value': 5,
    },
    {
        'name': '2.6 - Linear Momentum',
        'autolab': 'Linear Momentum Step',
        'handler': test_linear_momentum,
        'value': 3,
    },
    # {
    #     'name': '2.6 - Linear->Batchnorm->ReLU->Linear->Batchnorm->ReLU XELoss Momentum (Train/Eval)',
    #     'autolab': 'Big Linear ReLU XELoss Momentum Train/Eval',
    #     'handler': test_big_linear_batchnorm_relu_xeloss_momentum,
    #     'value': 7,
    # },
    {
        'name': '2.6 - Linear->ReLU->Linear->ReLU XELoss Momentum (Train/Eval)',
        'autolab': 'Big Linear ReLU XELoss Momentum Train/Eval',
        'handler': test_big_linear_relu_xeloss_momentum,
        'value': 7,
    },
    
    {
        'name': '3 - MNIST (NOT AUTOGRADED, WILL BE 0!)',
        'autolab': 'MNIST',
        'handler': lambda: False,
        'value': 10,
    }
]
tests.reverse()

if __name__ == '__main__':
    np.random.seed(2020)
    run_tests(tests)

    # Printing notice that mnist is not included in autograde score
    total_score = sum([t['value'] for t in tests])
    actual_total = total_score - [t['value']
                                  for t in tests if 'MNIST' in t['name']][0]
    print("\n***********************************************************************")
    print("NOTE: LOCAL FULL SCORE IS {}/{}; MNIST (10 points) IS NOT AUTOGRADED!".format(actual_total, total_score))
    # print("NOTE: IF `Addition` SCORES 0/0 WITH NO ERRORS, IT IS WORKING!")
    print("***********************************************************************")
