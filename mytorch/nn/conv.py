import numpy as np

import mytorch.nn.functional as F
from mytorch.nn.module import Module
from mytorch.tensor import Tensor


class Conv1d(Module):
    """1-dimensional convolutional layer.
    See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html for explanations
    and ideas.

    Notes:
        - No, you won't need to implement Conv2d for this homework. Conv2d will be in the bonus assignment.
        - These input args are all that you'll need for this class. You can add more yourself later
          if you want, but not required.

    Args:
        in_channel (int): # channels in input (example: # color channels in image)
        out_channel (int): # channels produced by layer
        kernel_size (int): edge length of the kernel (i.e. 3x3 kernel <-> kernel_size = 3)
        stride (int): Stride of the convolution (filter)
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        # Initializing weights and bias (not a very good initialization strategy)
        weight = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        self.weight = Tensor(weight, requires_grad=True, is_parameter=True)

        bias = np.zeros(out_channel)
        #bias = np.random.randn(out_channel)
        self.bias = Tensor(bias, requires_grad=True, is_parameter=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_size)
        Returns:
            Tensor: (batch_size, out_channel, output_size)
        """
        return F.Conv1d.apply(x, self.weight, self.bias, self.stride)
    

class Flatten(Module):
    """Layer that flattens all dimensions for each observation in a batch
    
    >>> x = torch.randn(4, 3, 2) # batch of 4 observations, each sized (3, 2)
    >>> x
    tensor([[[ 0.8816,  0.9773],
             [-0.1246, -0.1373],
             [-0.1889,  1.6222]],
             
            [[-0.9503, -0.8294],
             [ 0.8900, -1.2003],
             [-0.9701, -0.4436]],
             
            [[ 1.7809, -1.2312],
             [ 1.0769,  0.6283],
             [ 0.4997, -1.7876]],
             
            [[-0.5303,  0.3655],
             [-0.7496,  0.6935],
             [-0.8173,  0.4346]]]) 
    >>> layer = Flatten()
    >>> out = layer(x)
    >>> out 
    tensor([[ 0.8816,  0.9773, -0.1246, -0.1373, -0.1889,  1.6222],
            [-0.9503, -0.8294,  0.8900, -1.2003, -0.9701, -0.4436],
            [ 1.7809, -1.2312,  1.0769,  0.6283,  0.4997, -1.7876],
            [-0.5303,  0.3655, -0.7496,  0.6935, -0.8173,  0.4346]])
    >>> out.shape
    torch.size([4, 6]) # batch of 4 observations, each flattened into 1d array size (6,)
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, dim_2, dim_3, ...) arbitrary number of dims after batch_size 
        Returns:
            out (Tensor): (batch_size, dim_2 * dim_3 * ...) batch_size, then all other dims flattened
        """
        #s = np.prod(x.data.shape)
        #shape_1 = int(s/x.shape[0])
        shape2 = 1
        if len(x.data.shape) > 2:
            shape1 = x.data.shape[0]
            shape2 = int((np.prod(x.data.shape))/x.shape[0])
        else:
            shape1 = int(np.prod(x.data.shape))
            
        x = x.reshape(shape1,shape2)
        return x
        #raise Exception("TODO! One line of code. See writeup for hint.")
