import sys
import numpy as np
from torch import nn

sys.path.append('autograder')
sys.path.append('./')
sys.path.append('handin')

from test_util import *
from mytorch import tensor
from mytorch.nn.gru import GRUUnit, GRU
from mytorch.nn.util import pack_sequence as mpack_sequence

eps = 1e-6


def transfer_weights_GRU(src,dest,hs):
    
    i = 0 

    dest.unit.weight_ir.data = getattr(src,'weight_ih_l{}'.format(i))[:hs].detach().numpy()
    dest.unit.weight_iz.data = getattr(src,'weight_ih_l{}'.format(i))[hs:2*hs].detach().numpy()
    dest.unit.weight_in.data = getattr(src,'weight_ih_l{}'.format(i))[2*hs:].detach().numpy()
    
    dest.unit.bias_ir.data = getattr(src,'bias_ih_l{}'.format(i))[:hs].detach().numpy()
    dest.unit.bias_iz.data = getattr(src,'bias_ih_l{}'.format(i))[hs:2*hs].detach().numpy()
    dest.unit.bias_in.data = getattr(src,'bias_ih_l{}'.format(i))[2*hs:].detach().numpy()


    dest.unit.weight_hr.data = getattr(src,'weight_hh_l{}'.format(i))[:hs].detach().numpy()
    dest.unit.weight_hz.data = getattr(src,'weight_hh_l{}'.format(i))[hs:2*hs].detach().numpy()
    dest.unit.weight_hn.data = getattr(src,'weight_hh_l{}'.format(i))[2*hs:].detach().numpy()
    
    dest.unit.bias_hr.data = getattr(src,'bias_hh_l{}'.format(i))[:hs].detach().numpy()
    dest.unit.bias_hz.data = getattr(src,'bias_hh_l{}'.format(i))[hs:2*hs].detach().numpy()
    dest.unit.bias_hn.data = getattr(src,'bias_hh_l{}'.format(i))[2*hs:].detach().numpy()

def transfer_weights_gru_unit(src, dest, hs):
    dest.weight_ir.data = getattr(src,'weight_ih')[:hs].detach().numpy()
    dest.weight_iz.data = getattr(src,'weight_ih')[hs:2*hs].detach().numpy()
    dest.weight_in.data = getattr(src,'weight_ih')[2*hs:].detach().numpy()
        
    dest.bias_ir.data = getattr(src,'bias_ih')[:hs].detach().numpy()
    dest.bias_iz.data = getattr(src,'bias_ih')[hs:2*hs].detach().numpy()
    dest.bias_in.data = getattr(src,'bias_ih')[2*hs:].detach().numpy()

    dest.weight_hr.data = getattr(src,'weight_hh')[:hs].detach().numpy()
    dest.weight_hz.data = getattr(src,'weight_hh')[hs:2*hs].detach().numpy()
    dest.weight_hn.data = getattr(src,'weight_hh')[2*hs:].detach().numpy()
        
    dest.bias_hr.data = getattr(src,'bias_hh')[:hs].detach().numpy()
    dest.bias_hz.data = getattr(src,'bias_hh')[hs:2*hs].detach().numpy()
    dest.bias_hn.data = getattr(src,'bias_hh')[2*hs:].detach().numpy()


def test_gru_unit_forward():

    input_sizes, hidden_sizes, data_lens = get_params()

    for input_size, hidden_size, data_len in zip(input_sizes, hidden_sizes, data_lens):
        
        in_mytorch = tensor.Tensor.randn(data_len[0],input_size)
        in_torch = get_same_torch_tensor(in_mytorch) 

        model_mytorch = GRUUnit(input_size, hidden_size)
        model_torch = nn.GRUCell(input_size, hidden_size).double()
        
        transfer_weights_gru_unit(model_torch,model_mytorch,hidden_size)
        
        resm = model_mytorch(in_mytorch)
        rest = model_torch(in_torch)
        
        assert check_val(resm, rest, eps = eps)

    return True

def test_gru_unit_backward():
    import pdb;
    #pdb.set_trace()

    input_sizes, hidden_sizes, data_lens = get_params()

    for input_size, hidden_size, data_len in zip(input_sizes, hidden_sizes, data_lens):
        
        in_mytorch = tensor.Tensor.randn(data_len[0],input_size)
        in_torch = get_same_torch_tensor(in_mytorch) 
        
        in_mytorch.requires_grad = True
        in_torch.requires_grad = True

        model_mytorch = GRUUnit(input_size, hidden_size)
        model_torch = nn.GRUCell(input_size, hidden_size).double()
        transfer_weights_gru_unit(model_torch,model_mytorch,hidden_size)
        
        resm = model_mytorch(in_mytorch)
        rest = model_torch(in_torch)
        
        lm = (resm**2).sum()
        lt = (rest**2).sum()
        
        lm.backward()
        lt.backward()
        
        assert compare_gru_unit_param_grad(model_torch, model_mytorch, hidden_size)
        assert check_grad(resm, rest, eps = eps)

    return True


def test_GRU_forward():
#def test_time_iterator_forward():

    input_sizes, hidden_sizes, data_lens = get_params()

    for input_size, hidden_size, data_len in zip(input_sizes, hidden_sizes, data_lens):
        
        seq_mytorch = [tensor.Tensor.randn(data_len[i],input_size) for i in range(len(data_len)) ]

        seq_torch = [get_same_torch_tensor(i) for i in seq_mytorch ]
        
        mpack = mpack_sequence(seq_mytorch)
        tpack = nn.utils.rnn.pack_sequence(seq_torch, enforce_sorted=False)

        model_mytorch = GRU(input_size, hidden_size)
        model_torch = nn.GRU(input_size, hidden_size, num_layers = 1, batch_first = False ).double()
        
        transfer_weights_GRU(model_torch, model_mytorch, hidden_size)
        
        resm, hm = model_mytorch(mpack)
        rest, ht = model_torch(tpack)
        
        assert check_val(resm.data, rest.data, eps = eps)
        
        t_idx = list(np.argsort(data_len)[::-1])

        # Torch returns data which can be bi-directional
        assert check_val(hm, ht[0][t_idx], eps = eps)

    return True
