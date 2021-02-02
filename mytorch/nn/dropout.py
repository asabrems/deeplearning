import mytorch.nn.functional as F
from mytorch.nn.module import Module


class Dropout(Module):
    """During training, randomly zeroes some input elements with prob `p`.
    
    This is done using a mask tensor with values sampled from a bernoulli distribution.
    The elements to zero are randomized on every forward call.
    
    Args:
        p (float): the probability that any neuron output is dropped
        
    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        return F.Dropout.apply(x, self.p, self.is_train)
