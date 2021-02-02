import numpy as np
from mytorch import tensor
from mytorch.tensor import Tensor
from mytorch.nn.module import Module
from mytorch.nn.activations import Tanh, ReLU, Sigmoid
from mytorch.nn.util import PackedSequence, pack_sequence, unpack_sequence


class RNNUnit(Module):
    '''
    This class defines a single RNN Unit block.

    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 
    '''

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh' ):
        
        super(RNNUnit,self).__init__()
        
        # Initializing parameters
        self.weight_ih = Tensor(np.random.randn(hidden_size,input_size), requires_grad=True, is_parameter=True)
        self.bias_ih   = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)
        self.weight_hh = Tensor(np.random.randn(hidden_size,hidden_size), requires_grad=True, is_parameter=True)
        self.bias_hh   = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.hidden_size = hidden_size
        
        # Setting the Activation Unit
        if nonlinearity == 'tanh':
            self.act = Tanh()
        elif nonlinearity == 'relu':
            self.act = ReLU()

    def __call__(self, input, hidden = None):
        return self.forward(input,hidden)

    def forward(self, input, hidden = None):
        '''
        Args:
            input (Tensor): (effective_batch_size,input_size)
            hidden (Tensor,None): (effective_batch_size,hidden_size)
        Return:
            Tensor: (effective_batch_size,hidden_size)
        '''
        effective_batch_size = input.shape[0]
        hidden_size = self.bias_hh.shape[0]
        """print(effective_batch_size, hidden_size)
        print("inner",self.weight_ih.shape)
        print("hidden",self.weight_hh.shape)
        print("bias",self.bias_hh.shape)"""
        if hidden is None:
            self.hidden = Tensor(np.zeros((effective_batch_size,hidden_size)))
        else:
            self.hidden = hidden
        # TODO: INSTRUCTIONS
        # Perform matrix operations to construct the intermediary value from input and hidden tensors
        # Apply the activation on the resultant
        # Remeber to handle the case when hidden = None. Construct a tensor of appropriate size, filled with 0s to use as the hidden.
        
        #raise NotImplementedError('Implement Forward')
        #trial = (self.weight_hh@self.hidden.T()).T()
        #print("trial",trial)
        self.hidden = self.act((self.weight_ih@input.T()).T() +self.bias_ih + (self.weight_hh@self.hidden.T()).T() + self.bias_hh)
        return self.hidden


class TimeIterator(Module):
    '''
    For a given input this class iterates through time by processing the entire
    seqeunce of timesteps. Can be thought to represent a single layer for a 
    given basic unit which is applied at each time step.  
    
    Args:
        basic_unit (Class): RNNUnit or GRUUnit. This class is used to instantiate the unit that will be used to process the inputs
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 

    '''

    def __init__(self, basic_unit, input_size, hidden_size, nonlinearity = 'tanh' ):
        super(TimeIterator,self).__init__()

        # basic_unit can either be RNNUnit or GRUUnit
        self.unit = basic_unit(input_size,hidden_size,nonlinearity)    

    def __call__(self, input, hidden = None):
        return self.forward(input,hidden)
    
    def forward(self,input,hidden = None):
        
        '''
        NOTE: Please get a good grasp on util.PackedSequence before attempting this.

        Args:
            input (PackedSequence): input.data is tensor of shape ( total number of timesteps (sum) across all samples in the batch, input_size)
            hidden (Tensor, None): (batch_size, hidden_size)
        Returns:
            PackedSequence: ( total number of timesteps (sum) across all samples in the batch, hidden_size)
            Tensor (batch_size,hidden_size): This is the hidden generated by the last time step for each sample joined together. Samples are ordered in descending order based on number of timesteps. This is a slight deviation from PyTorch.
        '''
        
        # Resolve the PackedSequence into its components
        data, sorted_indices, batch_sizes = input
        
        # TODO: INSTRUCTIONS
        # Iterate over appropriate segments of the "data" tensor to pass same timesteps across all samples in the batch simultaneously to the unit for processing.
        # Remeber to account for scenarios when effective_batch_size changes between one iteration to the next
        #raise NotImplementedError('Implement Forward')
        inn = 0
        outt = 0
        hidden_state_pile = []
        

        for c,i in enumerate(batch_sizes):
            outt = inn+i
            effective_batch = data[inn:outt]
            
            
            if(hidden is not None):
                
                if(len(effective_batch) < len(hidden)):
                    diff =len(effective_batch) - len(hidden)  # find the difference between effective batch and hidden value 
                    hidden = hidden[0:diff]
            hidden = self.unit(effective_batch, hidden)
        
            hidden_state_pile.append(hidden) 
            
            inn = outt
            
        hidden_state_pile1 = PackedSequence(tensor.cat(hidden_state_pile,dim = 0),sorted_indices, batch_sizes) 
        
        hidden_tensor = np.zeros((len(sorted_indices),hidden.shape[1]))

        for c,i in enumerate(hidden_state_pile):

            hidden_tensor[batch_sizes[c]-1] = i[-1].data        
        
        return hidden_state_pile1, Tensor(hidden_tensor)
    
            
            


class RNN(TimeIterator):
    '''
    Child class for TimeIterator which appropriately initializes the parent class to construct an RNN.
    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 
    '''

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh' ):
        # TODO: Properly Initialize the RNN class
         super().__init__(RNNUnit,input_size,hidden_size,nonlinearity)
        
        
        #raise NotImplementedError('Initialize properly!')

