from mytorch import tensor

def backward(grad_fn, grad_of_outputs):
    """Recursive DFS that traverses comp graph, handing back gradients as it goes.
    Args:
        grad_fn (BackwardFunction or AccumulateGrad): Current node type from
                                                      parent's `.next_functions`
        grad_of_output (Tensor): Gradient of the final node w.r.t. current output
    Returns:
        No return statement needed.
    """

    # 1) Calculate gradients of final node w.r.t. to the current nodes parents
    #pass

    # 2) Pass gradient onto current node's beloved parents (recursive DFS)
    #pass

    #..............................
    
    if(grad_fn is not None):
        grad_of_outputs = grad_fn.apply(grad_of_outputs)
        acc = 0
        for parent in grad_fn.next_functions:
             if(parent is not None):
                 if(type(grad_of_outputs) == tuple):
                     if(acc < len(grad_of_outputs)):
                            backward(parent,grad_of_outputs[acc])
                            acc = acc+1      
                 elif (type(grad_of_outputs).__name__  == "Tensor"):
                        backward(parent,grad_of_outputs)
           
    #.................................


class Function:
    """Superclass for linking nodes to the computational graph.
    Operations in `functional.py` should inherit from this"""
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("All subclasses must implement forward")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("All subclasses must implement backward")

    @classmethod
    def apply(cls, *args):
        """Runs forward of subclass and links node to the comp graph.
        Args:
            cls (subclass of Function): (NOTE: Don't provide this;
                                               already provided by `@classmethod`)
                                        Current function, such as Add, Sub, etc.
            args (tuple): arguments for the subclass's `.forward()`.
                  (google "python asterisk arg")
        Returns:
            Tensor: Output tensor from operation that stores the current node.
        """
        # Creates BackwardFunction obj representing the current node
        #backward_function = BackwardFunction(cls)

        # Run subclass's forward with context manager and operation input args
        #output_tensor = cls.forward(backward_function.ctx, *args)

        # TODO: Complete code below
        # 1) For each parent tensor in args, add their node to `backward_function.next_functions`
        #    Note: Parents may/may not already have their own nodes. How do we handle this?
        #    Note: Parents may not need to be connected to the comp graph. How do we handle this?
        #    (see Appendix A.1 for hints)


        # 2) Store current node in output tensor (see `tensor.py` for ideas)
        # TODO: Write code here
        #.............................
        backward_function = BackwardFunction(cls)
        output_tensor = cls.forward(backward_function.ctx, *args)
        
        for parent in args:
            #print("parents coming in",parent.shape)
            if(type(parent).__name__ == "Tensor"):
            # this is performing a forward function and stores args this calls the forward of the operation
                
                if((parent.is_leaf is True) and (parent.requires_grad is True)):
                    accumulate = AccumulateGrad(parent)
                    backward_function.next_functions.append(accumulate)
                    
                elif((parent.is_leaf is False) and (parent.requires_grad is True)):
                    backward_function.next_functions.append(parent.grad_fn) # should have the grad_fn value
                else:#((parent.is_leaf is True) and (parent.requires_grad is False)):
                    backward_function.next_functions.append(None)
            else:
                backward_function.next_functions.append(None)
                
        output_tensor.grad_fn = backward_function
        #...............................

        return output_tensor


class AccumulateGrad:
    """Represents node where gradient must be accumulated.
    Args:
        tensor (Tensor): The tensor where the gradients are accumulated in `.grad`
    """
    def __init__(self, tensor):
        self.variable = tensor
        self.next_functions = [] # nodes of current node's parents (this WILL be empty)
                                 # exists just to be consistent in format with BackwardFunction
        self.function_name = "AccumulateGrad" # just for convenience lol

    def apply(self, arg):
        """Accumulates gradient provided.
        (Hint: Notice name of function is the same as BackwardFunction's `.apply()`)
        Args:
            arg (Tensor): Gradient to accumulate
        """
        # if no grad stored yet, initialize. otherwise +=
       # if no grad stored yet, initialize. otherwise +=
        if self.variable.grad is None:
            self.variable.grad = tensor.Tensor(arg.data)
        else:
            self.variable.grad.data += arg.data

        # Some tests to make sure valid grads were stored.
        shape = self.variable.shape
        grad_shape = self.variable.grad.shape
        assert shape == grad_shape, (shape, grad_shape)

class ContextManager:
    """Used to pass variables between a function's `.forward()` and `.backward()`.
    (Argument "ctx" in these functions)

    To store a tensor:
    >>> ctx.save_for_backward(<tensors>, <to>, <store>)

    To store other variables (like integers):
    >>> ctx.<some_name> = <some_variable>
    """
    def __init__(self):
        self.saved_tensors = [] # list that TENSORS get stored in

    def save_for_backward(self, *args):
        """Saves TENSORS only
        See example above for storing other data types.
        Args:
            args (Tensor(s)): Tensors to store
        """
        for arg in args:
            # Raises error if arg is not tensor (i warned you)
            if type(arg).__name__ != "Tensor":
                raise Exception("Got type {} of object {}. \nOnly Tensors should be saved in save_for_backward. For saving constants, just save directly as a new attribute.".format(type(arg), arg))

            self.saved_tensors.append(arg.copy())


class BackwardFunction:
    """Representing an intermediate node where gradient must be passed.
    Stored on output tensor of operation during `Function.apply()`
    
    Args:
        cls (subclass of Function): Operation being run. Don't worry about this;
                                    already handled in `Function.apply()`
    """
    def __init__(self, cls):
        self.ctx = ContextManager() # Just in case args need to be passed (see above)
        self._forward_cls = cls

        # Nodes of parents, populated in `Function.apply`
        self.next_functions = []

        # The name of the operation as a string (for convenience)
        self.function_name = cls.__name__

    def apply(self, *args):
        """Generates gradient by running the operation's `.backward()`.
        Args:
            args: Args for the operation's `.backward()`
        Returns:
            Tensor: gradient of parent's output w.r.t. current output
        """
        # Note that we've already provided the ContextManager
        return self._forward_cls.backward(self.ctx, *args)
