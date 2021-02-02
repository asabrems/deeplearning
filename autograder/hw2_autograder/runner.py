#!/usr/bin/env python3

import sys

sys.path.append('hw2')

from test_scanning import *
from test_conv import *
from test_cnn import test_cnn_step
import argparse

sys.path.append('autograder')
from helpers import *


parser = argparse.ArgumentParser(description='Run the hw2p1 autograder')
parser.add_argument('-s', '--summarize', action='store_true',
                    help='only show summary of scores')
args = parser.parse_args()

################################################################################
#################################### DO NOT EDIT ###############################
################################################################################

tests = [

    {
        'name': 'Question 1.1.1 - Conv1D Forward',
        'autolab': 'Conv1D Forward',
        'handler': test_conv1d_forward,
        'value': 20,
    },
    {
        'name': 'Question 1.1.2 - Conv1D Backward',
        'autolab': 'Conv1D Backward',
        'handler': test_conv1d_backward,
        'value': 30,
    },
    {
        'name': 'Question 1.2 - Flatten',
        'autolab': 'Flatten',
        'handler': test_flatten,
        'value': 10,
    },
    {
        'name': 'Question 2.1 - CNN as a Simple Scanning MLP',
        'autolab': 'SimpleScanningMLP',
        'handler': test_simple_scanning_mlp,
        'value': 10,
    },
    {
        'name': 'Question 2.2 - CNN as a Distributed Scanning MLP',
        'autolab': 'DistributedScanningMLP',
        'handler': test_distributed_scanning_mlp,
        'value': 10,
    },
    {
        'name': 'Question 3 - CNN',
        'autolab': 'CNN',
        'handler': test_cnn_step,
        'value': 15,
    },
]

tests.reverse()

if __name__ == '__main__':
    np.random.seed(2020)
    run_tests(tests, summarize=args.summarize)