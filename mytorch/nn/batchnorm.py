from mytorch.tensor import Tensor
import numpy as np
from mytorch.nn.module import Module

class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """
        #raise Exception("TODO!")
        if self.is_train == True:
            self.num_features = Tensor(self.num_features)
            #print("type: ",type(self.num_features))
            mean = x.sum(axis = 0, keepdims = True)*(Tensor(1)/self.num_features)
          
            var =  ((x - mean)*(x - mean)).sum(axis = 0, keepdims = True)
            var = (Tensor(1)/self.num_features)*var
            
            new_var = (Tensor(1)/(self.num_features- Tensor(1)))*(((x -mean).square()).sum(axis = 0, keepdims = True))
            self.running_mean = ((Tensor(1)-self.momentum)*self.running_mean) +  (self.momentum*mean)
            self.running_var =  ((Tensor(1)-self.momentum)*self.running_var ) +  (self.momentum*new_var)
            
            x= (x- mean)/(var + self.eps).sqrt()
        
            
            x = ((self.gamma*x) + self.beta)
            
            return x
    
        elif self.is_train == False:
            x = (x-self.running_mean)/(self.running_var + self.eps).sqrt()
            print("...........second...............")
            x = self.gamma*x + self.beta
            
            return x
            
        #eturn output
            
            
        
        
        

        return output