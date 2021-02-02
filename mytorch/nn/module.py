from mytorch.tensor import Tensor

class Module:
    """Base class (superclass) for all components of an NN.
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    
    Layer classes and even full Model classes should inherit from this Module.
    Inheritance gives the subclass all the functions/variables below
    
    NOTE: You shouldn't ever need to instantiate Module() directly."""

    def __init__(self):
        self._submodules = {} # Submodules of the class
        self._parameters = {} # Trainable params in module and its submodules

        self.is_train = True # Indicator for whether or not model is being trained.

    def train(self):
        """Activates training mode for network component"""
        self.is_train = True

    def eval(self):
        """Activates evaluation mode for network component"""
        self.is_train = False

    def forward(self, *args):
        """Forward pass of the module"""
        raise NotImplementedError("Subclasses of Module must implement forward")

    def is_parameter(self, obj):
        """Checks if input object is a Tensor of trainable params"""
        return isinstance(obj, Tensor) and obj.is_parameter
    
    def parameters(self):
        """Returns an interator over stored params.
        Includes submodules' params too"""
        self._ensure_is_initialized()
        for name, parameter in self._parameters.items():
            yield parameter
        for name, module in self._submodules.items():
            for parameter in module.parameters():
                yield parameter

    def add_parameter(self, name, value):
        """Stores params"""
        self._ensure_is_initialized()
        self._parameters[name] = value

    def add_module(self, name, value):
        """Stores module and its params"""
        self._ensure_is_initialized()
        self._submodules[name] = value

    def __setattr__(self, name, value):
        """Magic method that stores params or modules that you provide"""
        if self.is_parameter(value):
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)

        object.__setattr__(self, name, value)
        
    def __call__(self, *args):
        """Runs self.forward(args). Google 'python callable classes'"""
        return self.forward(*args)

    def _ensure_is_initialized(self):
        """Ensures that subclass's __init__() method ran super().__init__()"""
        if self.__dict__.get('_submodules') is None:
            raise Exception("Module not intialized. "
                            "Did you forget to call super().__init__()?")
