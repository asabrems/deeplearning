import sys
import numpy as np
import torch
from torch import nn

sys.path.append('autograder')
sys.path.append('./')
sys.path.append('handin')
from mytorch.tensor import Tensor
from mytorch.nn.rnn import RNN
from mytorch.nn.rnn import RNNUnit, TimeIterator
from mytorch.nn.util import pack_sequence as mpack_sequence
from test_util import *

eps = 1e-6

def transfer_weights(src,dest):
    # Assuming src to be a pytorch model
    # Assuming dest to be a mytorch model
    
    i=0

    dest.unit.weight_ih.data = getattr(src,'weight_ih_l{}'.format(i)).detach().numpy()
    dest.unit.weight_hh.data = getattr(src,'weight_hh_l{}'.format(i)).detach().numpy()
    dest.unit.bias_ih.data = getattr(src,'bias_ih_l{}'.format(i)).detach().numpy()
    dest.unit.bias_hh.data = getattr(src,'bias_hh_l{}'.format(i)).detach().numpy()


def test_rnn_unit_forward():

    input_sizes, hidden_sizes, data_lens = get_params()

    for input_size, hidden_size, data_len in zip(input_sizes, hidden_sizes, data_lens):
        
        in_mytorch = Tensor.randn(data_len[0],input_size)
        in_torch = get_same_torch_tensor(in_mytorch) 

        model_mytorch = RNNUnit(input_size, hidden_size)
        model_torch = nn.RNNCell(input_size, hidden_size).double()
        transfer_weights_rnn_unit(model_torch, model_mytorch)
        
        resm = model_mytorch(in_mytorch)
        rest = model_torch(in_torch)
        
        assert check_val(resm, rest, eps = eps)

    return True

def test_rnn_unit_backward():

    input_sizes, hidden_sizes, data_lens = get_params()

    for input_size, hidden_size, data_len in zip(input_sizes, hidden_sizes, data_lens):
        
        in_mytorch = Tensor.randn(data_len[0],input_size)
        in_torch = get_same_torch_tensor(in_mytorch) 
        
        in_mytorch.requires_grad = True
        in_torch.requires_grad = True

        model_mytorch = RNNUnit(input_size, hidden_size)
        model_torch = nn.RNNCell(input_size, hidden_size).double()
        transfer_weights_rnn_unit(model_torch, model_mytorch)
        
        resm = model_mytorch(in_mytorch)
        rest = model_torch(in_torch)
        
        lm = (resm**2).sum()
        lt = (rest**2).sum()
        
        lm.backward()
        lt.backward()
        
        assert compare_rnn_unit_param_grad(model_torch, model_mytorch)
        assert check_grad(resm, rest, eps = eps)

    return True

def test_time_iterator_forward():

    input_sizes, hidden_sizes, data_lens = get_params()

    for input_size, hidden_size, data_len in zip(input_sizes, hidden_sizes, data_lens):
        
        seq_mytorch = [Tensor.randn(data_len[i],input_size) for i in range(len(data_len)) ]

        seq_torch = [get_same_torch_tensor(i) for i in seq_mytorch ]
        
        mpack = mpack_sequence(seq_mytorch)
        tpack = nn.utils.rnn.pack_sequence(seq_torch, enforce_sorted=False)

        model_mytorch = RNN(input_size, hidden_size)
        model_torch = nn.RNN(input_size, hidden_size, num_layers = 1, batch_first = False ).double()
        
        transfer_weights(model_torch, model_mytorch)
        
        resm, hm = model_mytorch(mpack)
        rest, ht = model_torch(tpack)
        
        assert check_val(resm.data, rest.data, eps = eps)
        
        t_idx = list(np.argsort(data_len)[::-1])

        # Torch returns data which can be bi-directional
        assert check_val(hm, ht[0][t_idx], eps = eps)

    return True

def test_time_iterator_backward():

    input_sizes, hidden_sizes, data_lens = get_params()

    for input_size, hidden_size, data_len in zip(input_sizes, hidden_sizes, data_lens):
        
        seq_mytorch = [Tensor.randn(data_len[i],input_size) for i in range(len(data_len)) ]

        seq_torch = [get_same_torch_tensor(i) for i in seq_mytorch ]
        
        mpack = mpack_sequence(seq_mytorch)
        tpack = nn.utils.rnn.pack_sequence(seq_torch, enforce_sorted=False)

        model_mytorch = RNN(input_size, hidden_size)
        model_torch = nn.RNN(input_size, hidden_size, num_layers = 1, batch_first = False ).double()
        
        transfer_weights(model_torch, model_mytorch)
        
        resm, hm = model_mytorch(mpack)
        rest, ht = model_torch(tpack)
        
        
        lm = (resm.data**2).sum()
        lt = (rest.data**2).sum()
    
        lm.backward()
        lt.backward()
        
        assert compare_rnn_param_grad(model_torch,model_mytorch)
    
        for ma, pa in zip(seq_mytorch,seq_torch):
            assert check_grad(ma,pa)

    return True
