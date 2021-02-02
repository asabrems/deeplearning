import os
import sys
import numpy as np

from mytorch.nn.activations import Tanh, ReLU, Sigmoid
from mytorch.nn.conv import Conv1d, Flatten
from mytorch.nn.functional import get_conv1d_output_size
from mytorch.nn.linear import Linear
from mytorch.nn.module import Module
from mytorch.nn.sequential import Sequential


class CNN(Module):
    """A simple convolutional neural network with the architecture described in Section 3.
    
    You'll probably want to implement `get_final_conv_output_size()` on the
        bottom of this file as well.
    """
    def __init__(self):
        super().__init__()
        
        # You'll need these constants for the first layer
        first_input_size = 60 # The width of the input to the first convolutional layer
        first_in_channel = 24 # The number of channels input into the first layer
        conv1 = Conv1d(first_in_channel, 56, 5, 1)
        conv2 = Conv1d(56, 28, 6, 2)
        conv3 = Conv1d(28, 14, 2, 2)
        
        # TODO: initialize all layers EXCEPT the last linear layersnipping
        layers = [
                conv1,
                Tanh(),
                conv2,
                ReLU(),
                conv3,
                Sigmoid(),
                Flatten()
            # ... etc ... put layers in here, comma separated
        ]
        # TODO: Iterate through the conv layers and calculate the final output size
        
        output_size = get_final_conv_output_size(layers, first_input_size) 
        in_feature = conv3.out_channel*output_size
        linear = Linear(in_feature, 10)  
        layers.append(linear)
        # TODO: Append the linear layer with the correct size onto `layers`
        
        # TODO: Put the layers into a Sequential
        self.layers = Sequential(*layers)#Conv1d(first_in_channel, 56, 5, 1),Tanh(),Conv1d(56, 28, 6, 2),ReLU(),Conv1d(28, 14, 2, 2),Sigmoid(),Flatten(), linear)#None
        
        

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channels, input_size)
        Return:
            out (np.array): (batch_size, out_feature)
        """
        # Already completed for you. Passes data through all layers in order.
        return self.layers(x)


def get_final_conv_output_size(layers, input_size):
    """Calculates how the last dimension of the data will change throughout a CNN model
    
    Note that this is the final output size BEFORE the flatten.
    
    Note:
        - You can modify this function to use in HW2P2.
        - If you do, consider making these changes:
            - Change `layers` variable to `model` (a subclass of `Module`),
                and iterate through its submodules
                - Make a toy model in torch first and learn how to do this
            - Modify calculations to account for other layer types (like `Linear` or `Flatten`)
            - Change `get_conv1d_output_size()` to account for stride and padding (see its comments)
    
    Args:
        layers (list(Module)): List of Conv1d layers, activations, and flatten layers
        input_size (int): input_size of x, the input data 
        
    """
    output_size = 0
    for i,layer in enumerate(layers):
        if isinstance(layer, Conv1d):
            output_size = get_conv1d_output_size(input_size, layer.kernel_size, layer.stride)
            input_size = output_size 
    return output_size
    # Hint, you may find the function `isinstance()` to be useful.
    #raise NotImplementedError("TODO: Complete get_final_conv_output_size()!")
