from mytorch.nn.module import Module

class Sequential(Module):
    """Passes input data through stored layers, in order

    >>> model = Sequential(Linear(2,3), ReLU())
    >>> model(x)
    <output after linear then relu>

    Inherits from:
        Module (nn.module.Module)
    """
    def __init__(self, *layers):
        #print("layers",layers)
        super().__init__()
        self.layers = layers

        # iterate through args provided and store them
        for idx, l in enumerate(self.layers):
            self.add_module(str(idx), l)

    def __iter__(self):
        """Enables list-like iteration through layers"""
        yield from self.layers

    def __getitem__(self, idx):
        """Enables list-like indexing for layers"""
        return self.layers[idx]

    def train(self):
        """Sets this object and all trainable modules within to train mode"""
        self.is_train = True
        for submodule in self._submodules.values():
            submodule.train()

    def eval(self):
        """Sets this object and all trainable modules within to eval mode"""
        self.is_train = False
        for submodule in self._submodules.values():
            submodule.eval()

    def forward(self, x):
        """Passes input data through each layer in order
        Args:
            x (Tensor): Input data
        Returns:
            Tensor: Output after passing through layers
        """
        for layer in self.layers:
            x = layer(x)
            
        return x 
        #raise Exception("TODO!") # loop through self.layers...

