
import numpy as np

from mytorch.nn.module import Module
from mytorch.tensor import Tensor
from mytorch.nn.functional import *

class Linear(Module):
    """A linear layer (aka 'fully-connected' or 'dense' layer)
    
    >>> layer = Linear(2,3)
    >>> layer(Tensor.ones(10,2)) # (batch_size, in_features)
    <some tensor output with size (batch_size, out_features)>
    
    Args:
        in_features (int): # dims in input
                           (i.e. # of inputs to each neuron)
        out_features (int): # dims of output
                           (i.e. # of neurons)
        
    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Randomly initializing layer weights
        k = 1 / in_features
        weight = k * (np.random.rand(out_features, in_features) - 0.5)
        bias = k * (np.random.rand(out_features) - 0.5)
        self.weight = Tensor(weight, requires_grad=True, is_parameter=True)
        self.bias = Tensor(bias, requires_grad=True, is_parameter=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, in_features)
        Returns:        
            Tensor: (batch_size, out_features)
        """
        linear = x@self.weight.T() + self.bias
        #linear = Tensor.add(Tensor.mul(x,Tensor.Transpose(self.weight)),self.bias)
        return linear
        #raise Exception("TODO: Implement forward by calling the operations you've made")
