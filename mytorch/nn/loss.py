import numpy as np

import mytorch.nn.functional as F
from mytorch.tensor import Tensor


class Loss():
    """Base class for loss functions."""
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError("Loss subclasses must implement forward")


class CrossEntropyLoss(Loss):
    """The XELoss function.
    This class is for human use; just calls function in nn.functional.
    Does not need args to initialize.
    
    >>> criterion = CrossEntropyLoss()
    >>> criterion(outputs, labels)
    3.241
    """
    def __init__(self):
        pass

    def forward(self, predicted, target):
        """
        Args:
            predicted (Tensor): (batch_size, num_classes)
            target (Tensor): (batch_size,)

        Returns:
            Tensor: loss, stored as a float in a tensor 
        """
        # Simply calls nn.functional.cross_entropy
        # If you implement your own Function subclass you may need to modify this
        return F.cross_entropy(predicted, target)
