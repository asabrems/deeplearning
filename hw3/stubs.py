"""This file contains new code for hw3 that you should copy+paste to the appropriate files.

We'll tell you where each method/class belongs."""



# ---------------------------------
# nn/functional.py
# ---------------------------------

class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        raise NotImplementedError('Implemented Slice.forward')

    @staticmethod
    def backward(ctx,grad_output):
        raise NotImplementedError('Implemented Slice.backward')


################### THE CONCATENATION #####################

### ALTERNATE 1

# This should work for everyone. The non-tesnor argument is brought to the end so that this become similar to the other functions such as Reshape in terms of what you return from backward and how you handle this in your Autograd engine

# ---------------------------------
# nn/functional.py
# ---------------------------------

class Cat(Function):
    @staticmethod
    def forward(ctx,*args):
        '''
        Args:
            args (list): [*seq, dim] 
        
        NOTE: seq (list of tensors) contains the tensors that we wish to concatenate while dim (int) is the dimension along which we want to concatenate 
        '''
        *seq, dim = args

        raise NotImplementedError('Implement Cat.forward')

    @staticmethod
    def backward(ctx,grad_output):
        raise NotImplementedError('Implement Cat.backward')


### ALTERNATE 2

# For people who are explicitly handling the Nones being returned from the backward functions, this implementaion should work. If you find yourself encounterin strange issues related to None during the Cat testing, please try the other alternative

# ---------------------------------
# nn/functional.py
# ---------------------------------

class Cat(Function):
    @staticmethod
    def forward(ctx,dim,*seq):
        '''
        Args:
            dim (int): The dimension along which we concatenate our tensors
            seq (list of tensors): list of tensors we wish to concatenate
        '''
        raise NotImplementedError('Implement Cat.forward')

    @staticmethod
    def backward(ctx,grad_output):
        # NOTE: Be careful about handling the None corresponding to dim that you return, in case your implementation expects that for non-tensors
        raise NotImplementedError('Implement Cat.backward')


# ---------------------------------
# tensor.py
# ---------------------------------

# NOTE: This is a METHOD to be implmented in tensor.Tensor class.

def __len__(self,):
    return len(self.data)


# ---------------------------------
# tensor.py
# ---------------------------------

# NOTE: This is an indepedent function and NOT A METHOD in tensor.Tensor class 
def cat(seq,dim=0):
    '''
    Concatenates the given sequence of seq tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    
    NOTE: If you are not sure what this operation does, please revisit Recitation 0.

    Args:
        seq (list of Tensors) - List of interegers to concatenate
        dim (int) - The dimension along which to concatenate
    Returns:
        Tensor - Concatenated tensor

    Example:

        seq
        [[[3 3 4 1]
          [0 3 1 4]],
         [[4 2 0 0]
          [3 0 4 0]
          [1 4 4 3]],
         [[3 2 3 1]]]
        
        tensor.cat(seq,0)
        [[3 3 4 1]
         [0 3 1 4]
         [4 2 0 0]
         [3 0 4 0]
         [1 4 4 3]
         [3 2 3 1]]
    '''
    # TODO: invoke the appropriate function from functional.py. One-liner; don't overthink
    raise NotImplementedError("TODO: Complete functional.Cat!")

    
# ---------------------------------
# tensor.py
# ---------------------------------

# NOTE: This is a METHOD to be implmented in tensor.Tensor class.
def __getitem__(self, key):
    # TODO: Implement the __getitem__ operation. Simply invoke the appropriate function from functional.py
    raise NotImplementedError('TODO: Implement functional.Slice')



# ---------------------------------
# tensor.py
# ---------------------------------

# NOTE: This is a METHOD to be implemented in tensor.Tensor class.
def unsqueeze(self,dim=0):
    """ 
    Returns a new tensor with a dimension of size one inserted at the specified position. 
    
    NOTE: If you are not sure what this operation does, please revisit Recitation 0.
    
    Example:
        a
        [[1 2 3]
         [4 5 6]]
        
        a.unsqueeze(0)
        [[[1 2 3]
          [4 5 6]]]
        
        a.unsqueeze(1)
        [[[1 2 3]]
        
         [[4 5 6]]]
        
        a.unsqueeze(2)
        [[[1]
          [2]
          [3]]
        
         [[4]
          [5]
          [6]]]
    """
    # TODO: Implement the unsqueeze operation
    raise NotImplementedError('Use existing functions in functional.py to implement this operation!')


