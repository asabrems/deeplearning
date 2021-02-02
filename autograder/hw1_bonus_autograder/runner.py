#!/usr/bin/env python3

import sys

sys.path.append('autograder')

from helpers import *
from test_adam import *
from test_dropout import *
from test_batchnorm import *

################################################################################
#################################### DO NOT EDIT ###############################
################################################################################

tests = [
    {
        'name': 'Question 1 - Linear XELoss Adam',
        'autolab': 'Linear Adam',
        'handler': test_linear_adam,
        'value': 1,
    },
    {
        'name': 'Question 1 - Linear->ReLU->Linear->ReLU XELoss Adam',
        'autolab': 'Big Linear ReLU XELoss Adam',
        'handler': test_big_model_adam,
        'value': 4,
    },
    {
        'name': 'Question 2.1 - Linear->ReLU->Dropout (Forward)',
        'autolab': 'Linear ReLU Dropout',
        'handler': test_dropout_forward,
        'value': 3,
    },
    {
        'name': 'Question 2.2 - Linear->ReLU->Dropout (Backward)',
        'autolab': 'Linear ReLU Dropout Backward',
        'handler': test_dropout_forward_backward,
        'value': 3,
    },
    {
        'name': 'Question 2.2 - Linear->ReLU->Dropout (x2) XELoss Adam',
        'autolab': 'Big Linear BN ReLU Dropout XELoss Adam',
        'handler': test_big_model_step,
        'value': 4,
    },
    {
        'name': 'Question 3 - Linear->Batchnorm->ReLU Forward (Train)',
        'autolab': 'Linear Batchnorm ReLU Forward (Train)',
        'handler': test_linear_batchnorm_relu_forward_train,
        'value': 1,
    },
    {
        'name': 'Question 3 - Linear->Batchnorm->ReLU Backward (Train)',
        'autolab': 'Linear Batchnorm ReLU Backward (Train)',
        'handler': test_linear_batchnorm_relu_backward_train,
        'value': 1,
    },
    {
        'name': 'Question 3 - Linear->Batchnorm->ReLU (Train/Eval)',
        'autolab': 'Linear Batchnorm ReLU Train/Eval',
        'handler': test_linear_batchnorm_relu_train_eval,
        'value': 2,
    },
    {
        'name': 'Question 3 - Linear->Batchnorm->ReLU->Linear->Batchnorm->ReLU (Train/Eval)',
        'autolab': 'Big Linear Batchnorm ReLU Train/Eval',
        'handler': test_big_linear_batchnorm_relu_train_eval,
        'value': 6,
    },
]
tests.reverse()


if __name__ == '__main__':
    np.random.seed(2020)
    run_tests(tests)