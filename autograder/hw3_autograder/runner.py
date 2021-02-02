#!/usr/bin/env python3

import sys

sys.path.append('autograder')
sys.path.append('hw3')

from helpers import *
from test_gru import *
from test_rnn import *
from test_functional import *

import argparse

parser = argparse.ArgumentParser(description='Run the hw3p1 autograder')
parser.add_argument('-s', '--summarize', action='store_true',
                    help='only show summary of scores')
args = parser.parse_args()

################################################################################
#################################### DO NOT EDIT ###############################
################################################################################

tests = [
    {
        'name': '1.1.1 - RNN Unit Forward',
        'autolab': 'rnnunitforward',
        'handler': test_rnn_unit_forward,
        'value': 10,
    },
    {
        'name': '1.1.2 - RNN Unit Backward',
        'autolab': 'rnnunitbackward',
        'handler': test_rnn_unit_backward,
        'value': 5,
    },
    {
        'name': '1.2.1.1 - Concatenation Forward Test',
        'autolab': 'testcatforward',
        'handler': test_concat_forward,
        'value': 3,
    },
    {
        'name': '1.2.1.2 - Concatenation Backward Test',
        'autolab': 'testcatbackward',
        'handler': test_concat_backward,
        'value': 4,
    },
    {
        'name': '1.2.2.1 - Slicing Forward Test',
        'autolab': 'testsliceforward',
        'handler': test_slice_forward,
        'value': 2,
    },
    {
        'name': '1.2.2.2 - Slicing Backward Test',
        'autolab': 'testslicebackward',
        'handler': test_slice_backward,
        'value': 3,
    },
    {
        'name': '1.2.3 - Unsqueeze Test',
        'autolab': 'testunsqueeze',
        'handler': test_unsqueeze,
        'value': 3,
    },
    {
        'name': '1.3.1.1 - Pack Sequence Forward',
        'autolab': 'test_packseq_forward',
        'handler': test_pack_sequence_forward,
        'value': 10,
    },
    {
        'name': '1.3.1.2 - Pack Sequence Backward',
        'autolab': 'test_packseq_backward',
        'handler': test_pack_sequence_backward,
        'value': 5,
    },
    {
        'name': '1.3.2.1 - UnPack Sequence Forward',
        'autolab': 'test_unpackseq_forward',
        'handler': test_unpack_sequence_forward,
        'value': 10,
    },
    {
        'name': '1.3.2.2 - UnPack Sequence Backward',
        'autolab': 'test_unpackseq_backward',
        'handler': test_unpack_sequence_backward,
        'value': 5,
    },
    {
        'name': '1.4.1 - RNN Time Iterator Forward',
        'autolab': 'test_timeitr_forward',
        'handler': test_time_iterator_forward,
        'value': 10,
    },
    {
        'name': '1.4.2 - RNN Time Iterator Backward',
        'autolab': 'test_timeitr_backward',
        'handler': test_time_iterator_backward,
        'value': 10,
    },
    {
        'name': '2.1.1 - GRU Unit Forward',
        'autolab': 'test_gru_unit_forward',
        'handler': test_gru_unit_forward,
        'value': 10,
    },
    {
        'name': '2.1.2 - GRU Unit Backward',
        'autolab': 'test_gru_unit_backward',
        'handler': test_gru_unit_backward,
        'value': 5,
    },
    {
        'name': '2.2 - GRU Time Iterator Forward',
        'autolab': 'test_gru_forward',
        'handler': test_GRU_forward,
        'value': 5,
    }
]

tests.reverse()

if __name__ == '__main__':
    np.random.seed(2020)
    run_tests(tests, summarize=args.summarize)
