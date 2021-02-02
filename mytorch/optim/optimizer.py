import numpy as np


class Optimizer():
    """Base class for optimizers. Shouldn't need to modify."""
    def __init__(self, params):
        self.params = list(params)
        self.state = [] # Technically supposed to be a dict in real torch

    def step(self):
        """Called after generating gradients; updates network weights."""
        raise NotImplementedError

    def zero_grad(self):
        """After stepping, you need to call this to reset gradients.
        Otherwise they keep accumulating."""
        for param in self.params:
            param.grad = None
