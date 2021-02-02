import sys
import numpy as np
import torch
import math
from torch.nn.utils.rnn import pack_sequence

sys.path.append('autograder')

from helpers import assertions_all, assertions_no_type

sys.path.append('./')
sys.path.append('handin')

from mytorch.tensor import Tensor as mtensor
from mytorch.nn import util as mutil
from mytorch.nn.functional import *

def compare_ndarrays(a1,a2,eps=1e-8,test_name=''):
    # a1 is ndarray
    # a2 is PyTorch Tensor
    a2 = a2.detach().numpy()
    
    return assertions_all(a1,a2,test_name)    

def compare_ps(p1,p2, test_name):
    # Assuming p1 belongs to pytorch and p2 belongs to mytorch
    a1 = p1.data.detach().numpy()
    a2 = p2.data
    #print(np.abs(a1-a2).sum())
    assert assertions_all(a1, a2, test_name), "Torch and MyTorch results don't match."

def compare_ps_tensor(p1,p2, test_name):
    # Assuming p1 belongs to pytorch and p2 belongs to mytorch
    a1 = p1.data
    a2 = p2.data

    assert assertions_all(a1, a2, test_name), "Torch and MyTorch results don't match."

def get_params():
    input_sizes = [2,2,2,3,4]
    hidden_sizes = [2,2,4,3,3]
    
    data_lens = [[1],
                 [4],
                 [1,2,3],
                 [4,3,2],
                 [5,2,3]]

    return input_sizes, hidden_sizes, data_lens 

def get_torch_data(data):
    ##### Torch-based #####
    tx = [ torch.tensor(el).float() for el in data ]
    tx_pack = pack_sequence(tx,enforce_sorted=False)

    return tx_pack.data

def get_torch_pack(data):
    tx = [ torch.tensor(el).float() for el in data ]
    tx_pack = pack_sequence(tx,enforce_sorted=False)

    return tx_pack

def get_hidden_data(s1, s2):
    hidden = torch.tensor(np.zeros([s1,s2]))

    return hidden

def get_mytorch_data(data):
    ##### mytorch-based #####

    tx = [ torch.tensor(el).float() for el in data ]
    tx_pack = pack_sequence(tx,enforce_sorted=False)

    return tensor.Tensor(tx_pack.data.detach().numpy())

def get_mytorch_pack(data):
    ##### mytorch-based #####

    mx = [ mtensor(el) for el in data ]
    mx_pack = mutil.pack_sequence(mx)

    return mx_pack

def get_same_torch_tensor(mytorch_tensor):
    res = torch.tensor(mytorch_tensor.data).double()
    res.requires_grad = mytorch_tensor.requires_grad
    return res

def get_same_torch_tensor_float(mytorch_tensor):
    res = torch.tensor(mytorch_tensor.data).float()
    res.requires_grad = mytorch_tensor.requires_grad
    return res

def check_val_and_grad(mytorch_tensor, pytorch_tensor):
    return check_val(mytorch_tensor, pytorch_tensor) and \
           check_grad(mytorch_tensor, pytorch_tensor)

def check_val(mytorch_tensor, pytorch_tensor, eps = 9e-6):
    if not isinstance(pytorch_tensor, torch.DoubleTensor):
        #print("Warning: torch tensor is not a DoubleTensor. It is instead {}".format(pytorch_tensor.type()))
        #print("It is highly recommended that similarity testing is done with DoubleTensors as numpy arrays have 64-bit precision (like DoubleTensors)")
        pass
    if tuple(mytorch_tensor.shape) != tuple(pytorch_tensor.shape):
        print("mytorch tensor and pytorch tensor has different shapes: {}, {}".format(
            mytorch_tensor.shape, pytorch_tensor.shape
        ))
        return False
    
    data_diff = np.abs(mytorch_tensor.data - pytorch_tensor.data.numpy())
    max_diff = data_diff.max()
    if max_diff < eps:
        return True
    else:
        print("Data element differs by {}:".format(max_diff))
        print("mytorch tensor:")
        print(mytorch_tensor)
        print("pytorch tensor:")
        print(pytorch_tensor)

        return False

def check_grad(mytorch_tensor, pytorch_tensor, eps = 9e-6):
    if mytorch_tensor.grad is None or pytorch_tensor_nograd(pytorch_tensor):
        if mytorch_tensor.grad is None and pytorch_tensor_nograd(pytorch_tensor):
            return True
        elif mytorch_tensor.grad is None:
            print("Mytorch grad is None, but pytorch is not")
            return False
        else:
            print("Pytorch grad is None, but mytorch is not")
            return False

    grad_diff = np.abs(mytorch_tensor.grad.data - pytorch_tensor.grad.data.numpy())
    max_diff = grad_diff.max()
    if max_diff < eps:
        return True
    else:
        print("your grad",mytorch_tensor.grad.data)
        print("required grad",pytorch_tensor.grad.data.numpy())
        print("Grad differs by {}".format(grad_diff))
        return False

def pytorch_tensor_nograd(pytorch_tensor):
    return not pytorch_tensor.requires_grad or not pytorch_tensor.is_leaf


def compare_gru_unit_param_grad(src, dest, hs):
    a = check_val(dest.weight_ir.grad, getattr(src,'weight_ih').grad[:hs])
    a = a and check_val(dest.weight_iz.grad, getattr(src,'weight_ih').grad[hs:2*hs])
    a = a and check_val(dest.weight_in.grad, getattr(src,'weight_ih').grad[2*hs:])
        
    a = a and check_val(dest.bias_ir.grad, getattr(src,'bias_ih').grad[:hs])
    a = a and check_val(dest.bias_iz.grad, getattr(src,'bias_ih').grad[hs:2*hs])
    a = a and check_val(dest.bias_in.grad, getattr(src,'bias_ih').grad[2*hs:])

    a = a and check_val(dest.weight_hr.grad,getattr(src,'weight_hh').grad[:hs])
    a = a and check_val(dest.weight_hz.grad,getattr(src,'weight_hh').grad[hs:2*hs])
    a = a and check_val(dest.weight_hn.grad, getattr(src,'weight_hh').grad[2*hs:])
        
    a = a and check_val(dest.bias_hr.grad, getattr(src,'bias_hh').grad[:hs])
    a = a and check_val(dest.bias_hz.grad, getattr(src,'bias_hh').grad[hs:2*hs])
    a = a and check_val(dest.bias_hn.grad, getattr(src,'bias_hh').grad[2*hs:])
    
    return a

def compare_rnn_unit_param_grad(src, dest):
    a = check_grad(dest.weight_ih, getattr(src,'weight_ih'))
    a = a and check_grad(dest.weight_hh, getattr(src,'weight_hh'))
    a = a and check_grad(dest.bias_ih, getattr(src,'bias_ih'))
    a = a and check_grad(dest.bias_hh, getattr(src,'bias_hh'))
    
    return a

def transfer_weights_rnn_unit(src, dest):
    dest.weight_ih.data = getattr(src,'weight_ih').detach().numpy()
    dest.weight_hh.data = getattr(src,'weight_hh').detach().numpy()
    dest.bias_ih.data = getattr(src,'bias_ih').detach().numpy()
    dest.bias_hh.data = getattr(src,'bias_hh').detach().numpy()

def compare_rnn_param_grad(src,dest):
    # Assuming src to be a pytorch model
    # Assuming dest to be a mytorch model
    
    i=0

    a = check_grad(dest.unit.weight_ih, getattr(src,'weight_ih_l{}'.format(i)))
    a = a and check_grad(dest.unit.weight_hh, getattr(src,'weight_hh_l{}'.format(i)))
    a = a and check_grad(dest.unit.bias_ih, getattr(src,'bias_ih_l{}'.format(i)))
    a = a and check_grad(dest.unit.bias_hh, getattr(src,'bias_hh_l{}'.format(i)))
    
    return a

def transfer_weights_rnn(src,dest):
    # Assuming src to be a pytorch model
    # Assuming dest to be a mytorch model
    
    num_layers = len(dest.layers.sequence)
    bidirectional = (dest.num_directions == 2)
    
    for i in range(num_layers):
        if bidirectional:
            dest.layers_rev[i].unit.weight_ih.data = getattr(src,'weight_ih_l{}_reverse'.format(i)).detach().numpy()
            dest.layers_rev[i].unit.weight_hh.data = getattr(src,'weight_hh_l{}_reverse'.format(i)).detach().numpy()
            dest.layers_rev[i].unit.bias_ih.data = getattr(src,'bias_ih_l{}_reverse'.format(i)).detach().numpy()
            dest.layers_rev[i].unit.bias_hh.data = getattr(src,'bias_hh_l{}_reverse'.format(i)).detach().numpy()


        dest.layers[i].unit.weight_ih.data = getattr(src,'weight_ih_l{}'.format(i)).detach().numpy()
        dest.layers[i].unit.weight_hh.data = getattr(src,'weight_hh_l{}'.format(i)).detach().numpy()
        dest.layers[i].unit.bias_ih.data = getattr(src,'bias_ih_l{}'.format(i)).detach().numpy()
        dest.layers[i].unit.bias_hh.data = getattr(src,'bias_hh_l{}'.format(i)).detach().numpy()

def transfer_weights_linear(pModel,mModel):
    
    mModel.weight.data = pModel.weight.detach().numpy()
    mModel.bias.data = pModel.bias.detach().numpy()
    
def transfer_weights(pModel,mModel):
    
    transfer_weights_rnn(pModel.rnn,mModel.rnn)
    transfer_weights_linear(pModel.linear,mModel.linear)
