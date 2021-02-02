import sys
import numpy as np

from mytorch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    
    >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    Args:
        params (dict): <some module>.parameters()
        lr (float): learning rate (eta)
        momentum (float): momentum factor (beta)

    Inherits from:
        Optimizer (optim.optimizer.Optimizer)
    """
    def __init__(self, params, lr=0.001, momentum=0.0):
        super().__init__(params) # inits parent with network params
        self.lr = lr
        self.momentum = momentum

        # This tracks the momentum of each weight in each param tensor
        self.momentums = [np.zeros(t.shape) for t in self.params]

    def step(self):
        """Updates params based on gradients stored in param tensors"""
        for i in range(0, len(self.params)):
          
            #self.params[i].data =  self.params[i].data - self.lr*self.params[i].grad.data
            #print(self.params)
            #if(self.params[i].grad is not None):
            self.momentums[i] = self.momentum*self.momentums[i] - self.lr*(self.params[i].grad.data)
            self.params[i].data = self.params[i].data + self.momentums[i]
        #raise Exception("TODO: Implement SGD.step()")
