import sys
import numpy as np
import torch
import math
from test_util import *
from helpers import assertions_all

sys.path.append('autograder')
sys.path.append('./')
sys.path.append('handin')

from mytorch.autograd_engine import *
from mytorch.tensor import Tensor, cat
from mytorch.nn.util import pack_sequence, unpack_sequence

eps = 1e-6

def test_concat_forward():
    # shape of tensor to test
    tensor_shapes = [[(1,4,3),(1,8,3),(1,5,3)],
                     [(2,3,4),(1,3,4)],
                     [(6,7,8,9),(6,7,8,1),(6,7,8,2)],
                     [(1,2,3),(1,2,4),(1,2,3),(1,2,4)]]
    
    cat_dims = [1,0,3,2]

    for tensor_shapes_cur, d_cur in zip(tensor_shapes,cat_dims):
        # get mytorch and torch tensor: 'a'
        a = [ Tensor.randn(*shape_i) for shape_i in tensor_shapes_cur  ]
        for i in range(len(a)):
            a[i].requires_grad = True
        
        a_torch = [get_same_torch_tensor(a_i) for a_i in a ]

        c = cat(a, d_cur)
        c_torch = torch.cat(a_torch, dim=d_cur)
        
        assert check_val(c,c_torch, eps = eps)

    return True

def test_concat_backward():
    # shape of tensor to test
    tensor_shapes = [[(1,4,3),(1,8,3),(1,5,3)],
                     [(2,3,4),(1,3,4)],
                     [(6,7,8,9),(6,7,8,1),(6,7,8,2)],
                     [(1,2,3),(1,2,4),(1,2,3),(1,2,4)]]
    
    cat_dims = [1,0,3,2]

    for tensor_shapes_cur, d_cur in zip(tensor_shapes,cat_dims):
        # get mytorch and torch tensor: 'a'
        a = [ Tensor.randn(*shape_i) for shape_i in tensor_shapes_cur  ]
        for i in range(len(a)):
            a[i].requires_grad = True
        
        a_torch = [get_same_torch_tensor(a_i) for a_i in a ]

        c = cat(a, d_cur)
        c_torch = torch.cat(a_torch, dim=d_cur)
        
        l = (c**2).sum()
        l_torch = (c_torch**2).sum()
        
        l.backward()
        l_torch.backward()

        for a_i, a_torch_i in zip(a,a_torch):
            assert check_grad(a_i, a_torch_i, eps = eps)

    return True

def test_slice_forward():
    # shape of tensor to test
    shape = (2, 4, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape) #mytorch tensor
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a) #pytorch tensor

    # Test 1
    b = a[1,2,0]
    b_torch = a_torch[1,2,0]
    assert check_val(b,b_torch, eps = eps)

    # Test 2
    b = a[0,2,:]
    b_torch = a_torch[0,2,:]
    assert check_val(b,b_torch, eps = eps)

    # Test 3
    b = a[:,3,:]
    b_torch = a_torch[:,3,:]
    assert check_val(b,b_torch, eps = eps)

    # Test 4
    b = a[:,:,1]
    b_torch = a_torch[:,:,1]
    assert check_val(b,b_torch, eps = eps)

    # Test 5
    b = a[:,1:3,:]
    b_torch = a_torch[:,1:3,:]
    assert check_grad(b,b_torch, eps = eps)
    
    return True

def test_slice_backward():
    # shape of tensor to test
    shape = (2, 4, 3)

    # Test 1
    a = Tensor.randn(*shape) #mytorch tensor
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a) #pytorch tensor
    b = a[1,2,0]
    b_torch = a_torch[1,2,0]
    (b**2).sum().backward()
    (b_torch**2).sum().backward()
    assert check_grad(a,a_torch, eps = eps)

    # Test 2
    a = Tensor.randn(*shape) #mytorch tensor
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a) #pytorch tensor
    b = a[0,2,:]
    b_torch = a_torch[0,2,:]
    (b**2).sum().backward()
    (b_torch**2).sum().backward()
    assert check_grad(a,a_torch, eps = eps)

    # Test 3
    a = Tensor.randn(*shape) #mytorch tensor
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a) #pytorch tensor
    b = a[:,3,:]
    b_torch = a_torch[:,3,:]
    (b**2).sum().backward()
    (b_torch**2).sum().backward()
    assert check_grad(a,a_torch, eps = eps)

    # Test 4
    a = Tensor.randn(*shape) #mytorch tensor
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a) #pytorch tensor
    b = a[:,:,1]
    b_torch = a_torch[:,:,1]
    (b**2).sum().backward()
    (b_torch**2).sum().backward()
    assert check_grad(a,a_torch, eps = eps)

    # Test 5
    a = Tensor.randn(*shape) #mytorch tensor
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a) #pytorch tensor
    b = a[:,1:3,:]
    b_torch = a_torch[:,1:3,:]
    (b**2).sum().backward()
    (b_torch**2).sum().backward()
    assert check_grad(a,a_torch, eps = eps)
    

    return True

def test_unsqueeze():
    # shape of tensor to test
    shape = (2, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape) #mytorch tensor
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a) #pytorch tensor

    b = Tensor.unsqueeze(a)
    b_torch = torch.Tensor.unsqueeze(a_torch, 0)
    assert check_val(b, b_torch,eps=eps)
    
    b = Tensor.unsqueeze(a, 2)
    b_torch = torch.Tensor.unsqueeze(a_torch, 2)
    assert check_val(b, b_torch,eps=eps)
    
    return True

def test_pack_sequence_forward():
    
    test_shapes = [[(4,1),(5,1)],
                   [(4,3),(10,3),(2,3)]]
    
    a = True
    for shapes in test_shapes:
        # get mytorch and torch tensor: 'a'
        seq1 = [Tensor.randn(*shape) for shape in shapes]
        seq2 = [ get_same_torch_tensor(t) for t in seq1 ]
    
        # run mytorch and torch forward: 'c = cat (a, b)'
        c = pack_sequence(seq1)
        c_torch = torch.nn.utils.rnn.pack_sequence(seq2, enforce_sorted=False)
        #print(c.data)
        assert check_val(c.data,c_torch.data)
        #compare_ps(c_torch, c.data, "test_pack_sequence_forward")
        assert compare_ndarrays(c.batch_sizes, c_torch.batch_sizes, test_name = 'Testing batch_sizes')
        assert compare_ndarrays(c.sorted_indices, c_torch.sorted_indices, test_name = 'Testing sorted_indices')
    
    return True 


def test_pack_sequence_backward():
    
    test_shapes = [[(4,1),(5,1)],
                   [(4,3),(10,3),(2,3)]]
    
    a = True
    for shapes in test_shapes:
        # get mytorch and torch tensor: 'a'
        seq1 = [Tensor.randn(*shape) for shape in shapes]
        for t in seq1:
            t.requires_grad = True

        seq2 = [ get_same_torch_tensor(t) for t in seq1 ]
    
        # run mytorch and torch forward: 'c = cat (a, b)'
        c = pack_sequence(seq1)
        c_torch = torch.nn.utils.rnn.pack_sequence(seq2, enforce_sorted=False)
        
        l = (c.data**2).sum()
        l_torch = (c_torch.data**2).sum()

        l.backward()
        l_torch.backward()
        
        for a1, a2 in zip(seq1, seq2):
            assert check_grad(a1,a2, eps = eps)
        #compare_ps(c_torch, c.data, "test_pack_sequence_backward")
    
    return True


def test_unpack_sequence_forward():
    
    test_shapes = [[(4,1),(5,1),(2,1),(2,1),(3,1)],
                   [(4,3),(10,3),(2,3)]]

    for shapes in test_shapes:
        # get mytorch and torch tensor: 'a'
        seq1 = [Tensor.randn(*shape) for shape in shapes]
        # run mytorch and torch forward: 'c = cat (a, b)'
        c = pack_sequence(seq1)
        seq2 = unpack_sequence(c)
        
        for s1,s2 in zip(seq1,seq2): 
            assert assertions_all(s1.data, s2.data, 'Unpack Forward')
    
    return True 

def test_unpack_sequence_backward():
    
    test_shapes = [[(4,1),(5,1)],
                   [(4,3),(10,3),(2,3)]]
    a = True
    for shapes in test_shapes:
        # get mytorch and torch tensor: 'a'
        seq1 = [Tensor.randn(*shape) for shape in shapes]
        for t in seq1:
            t.requires_grad = True

        seq2 = [ get_same_torch_tensor(t) for t in seq1 ]
    
        # run mytorch and torch forward: 'c = cat (a, b)'
        c_temp = pack_sequence(seq1)
        c_temp2 = unpack_sequence(c_temp)
        c = pack_sequence(c_temp2)

        c_torch = torch.nn.utils.rnn.pack_sequence(seq2, enforce_sorted=False)
        
        l = (c.data**2).sum()
        l_torch = (c_torch.data**2).sum()

        l.backward()
        l_torch.backward()
        
        for a1, a2 in zip(seq1, seq2):
            assert check_grad(a1,a2, eps = eps)
        #compare_ps(c_torch, c.data, "test_pack_sequence_backward")
    
    return True
