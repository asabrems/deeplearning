import numpy as np
from scipy.stats import bernoulli
from mytorch import tensor
from mytorch.autograd_engine import Function

def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        print("shape in",shape)
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None





class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
       
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
                
        ctx.save_for_backward(a,b)
        
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,is_leaf=not requires_grad)
      
        
        return c
        #raise Exception("TODO: Implement '-' forward")

    @staticmethod
    def backward(ctx, grad_output):
        
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = -np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        
        grad_a = unbroadcast(grad_a, a.shape)
        grad_b = unbroadcast(grad_b, b.shape)
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

# TODO: Implement more Functions below

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        
                
        ctx.save_for_backward(a,b)
        
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor((a.data*b.data), requires_grad=requires_grad,is_leaf=not requires_grad)
      
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = b.data * grad_output.data #this is how i implemented mult
        grad_b = a.data * grad_output.data

      
        grad_a = unbroadcast(grad_a, a.shape)
        grad_b = unbroadcast(grad_b, b.shape)
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)
    
class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
            #raise Exception("TODO: Implement '-' forward")
        #ctx.save_for_backward(a, b)
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
    
                
        ctx.save_for_backward(a,b)
        
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor((a.data/b.data), requires_grad=requires_grad,is_leaf=not requires_grad)
        
    
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output.data/(b.data) #this is how i implemented mult
        grad_b = (-a.data*(grad_output.data))/(b.data**2)
        
        grad_a = unbroadcast(grad_a, a.shape)
        grad_b = unbroadcast(grad_b, b.shape)
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)
# TODO: Implement more Functions below
class Exp(Function):
    @staticmethod
    def forward(ctx, a):        
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__))

        # Check that args have same shape
        ctx.save_for_backward(a)
        
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad,is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a= ctx.saved_tensors[0]
        #print("size of a data:{}, b data:{} and grad_out{}".format(a.data.shape,b.data.shape,grad_output.shape))
       
        #grad_a = a.data*grad_output.data
        grad_a = np.ones(a.shape)*np.exp(a.data)*grad_output.data
        #print(grad_a.shape,grad_b.shape)
        
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a)

class Neg(Function):
    @staticmethod
    def forward(ctx, a):        
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__))

        # Check that args have same shape
        ctx.save_for_backward(a)
        
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.negative(a.data), requires_grad=requires_grad,is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a= ctx.saved_tensors[0]
        #print("size of a data:{}, b data:{} and grad_out{}".format(a.data.shape,b.data.shape,grad_output.shape))
       
        grad_a =-np.ones(a.shape)*grad_output.data
        
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a)

class Dot(Function):
    @staticmethod
    def forward(ctx, a, b):        
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        ctx.save_for_backward(a,b)
        
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.matmul(a.data,b.data), requires_grad=requires_grad,is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        #print("size of a data:{}, b data:{} and grad_out{}".format(a.data.shape,b.data.shape,grad_output.shape))
        if(len(grad_output.shape) == 1):
            grad_a =(np.matmul(b.data,np.transpose(grad_output.data)))
            grad_b = np.tensordot(a.data,grad_output.data, axes=0)
            
        else:
            grad_a = np.transpose(np.matmul(b.data,np.transpose(grad_output.data)))#this is how i implemented mult
            #grad_b = np.tensordot(a.data,grad_output.data, axes=0)
            #grad_b = np.matmul(np.transpose(np.broadcast_to(grad_output.data,b.shape)),a.data)
            grad_b = np.matmul(np.transpose(a.data),grad_output.data)
        #print(grad_a.shape,grad_b.shape)
        
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class RELU(Function):
    @staticmethod
    def forward(ctx, x):
        # Check that both args are tensors
        if not (type(x).__name__ == 'Tensor'):
            raise Exception("arg should be tensor: {} ".format(type(x).__name__))
            
        ctx.save_for_backward(x)

        requires_grad = x.requires_grad
        #print(x.data)
        x.data = x.data*(x.data > 0)
        #np.place(x.data,x.data<=0,0)
        c = tensor.Tensor(x.data, requires_grad=requires_grad,is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        
        x = ctx.saved_tensors[0]
       
        relu_der = (x.data> 0).astype(x.data.dtype)
        
        grad_x = relu_der*grad_output.data
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_x)
    
class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        #print(a.shape, c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape
   


    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss
    a = tensor.Tensor(np.max(predicted.data, axis = 1,keepdims=True))
    
    logSumExp = a + (predicted - a).exp().sum(axis = 1,keepdims=True).log()
    
    logSoftMax = (predicted - logSumExp)#+ tensor.Tensor(1)))
    
    target = to_one_hot(target,num_classes)
    
    nllloss = (target*logSoftMax).sum()/tensor.Tensor(batch_size)
    nllloss = nllloss.neg()
    
    return nllloss

def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]

    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)



"""def unbroadcast(x,shape):
  if(len(shape)==1):
    for i in range(0,len(x.shape)-1):
      x= x.sum(axis = 0)
      #x= np.array([x.sum(axis=0) for i in range(0,len(x.shape)-1)])
  else:
    for i in range(0,len(shape)):
      if(x.shape[i] != shape[i]):
        x = x.sum(axis =i)
  return x"""

class Sqrt(Function):
    @staticmethod
    def forward(ctx, x,y):
        # Check that both args are tensors
        if not (type(x).__name__ == 'Tensor'):
            raise Exception("arg should be tensor: {} ".format(type(x).__name__))
            
        ctx.save_for_backward(x)

        requires_grad = x.requires_grad
        #np.place(x.data,x.data<=0,0)
        c = tensor.Tensor(np.sqrt(x.data), requires_grad=requires_grad,is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        
        x = ctx.saved_tensors[0]
       
        grad_x = (1/(2*np.ones(x.shape)))*grad_output.data
        
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_x)

class Pow(Function):
    @staticmethod
    def forward(ctx, x,y):
        # Check that both args are tensors
        if not (type(x).__name__ == 'Tensor'):
            raise Exception("arg should be tensor: {} ".format(type(x).__name__))
            
        ctx.save_for_backward(x)

        requires_grad = x.requires_grad
        #np.place(x.data,x.data<=0,0)
        c = tensor.Tensor(x.data**2, requires_grad=requires_grad,is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        
        x = ctx.saved_tensors[0]
       
        grad_x = x.data*2*grad_output.data
        
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_x)

class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, is_train=False):
        """Forward pass for dropout layer.

        Args:
            ctx (ContextManager): For saving variables between forward and backward passes.
            x (Tensor): Data tensor to perform dropout on
            p (float): The probability of dropping a neuron output.
                       (i.e. 0.2 -> 20% chance of dropping)
            is_train (bool, optional): If true, then the Dropout module that called this
                                       is in training mode (`<dropout_layer>.is_train == True`).

                                       Remember that Dropout operates differently during train
                                       and eval mode. During train it drops certain neuron outputs.
                                       During eval, it should NOT drop any outputs and return the input
                                       as is. This will also affect backprop correspondingly.
        """       
        
        if not type(x).__name__ == 'Tensor':
            raise Exception("Only dropout for tensors is supported")
        x_1 = x.copy()
        if is_train == True:
            data_mask= 1- np.random.binomial(1, p, x.shape)
            
            x.data  = x.data*data_mask*(1/(1-p))
            value = x.data
            print(x)
            
        elif is_train == False:
            x.data = x_1.data
            value = x.data
            
        
        ctx.save_for_backward(x) 
        requires_grad = x.requires_grad
        
        c = tensor.Tensor(value, requires_grad=requires_grad,is_leaf=not requires_grad)
        
        return c
        #raise NotImplementedError("TODO: Implement Dropout(Function).forward() for hw1 bonus!")

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        print("value")
        print(x)
        print("..................yaay.....................")
        
        #print("grad_out type",type(grad_output))
        grad_x = (np.ones(x.shape)*grad_output.data)*x.data
        
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_x)
        
        #raise NotImplementedError("TODO: Implement Dropout(Function).backward() for hw1 bonus!")


#.....................................new chapter ..............................................#
def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.
        
        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    value = ((input_size -kernel_size)//stride) + 1 # dont forget to asdd math floor 
    return value
    #raise NotImplementedError("TODO: Complete functional.get_conv1d_output_size()!")
    

class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
       
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
      
        output_size =  get_conv1d_output_size(input_size, kernel_size, stride)
        
        out = np.zeros((batch_size,out_channel,output_size))
       
        ctx.save_for_backward(x,tensor.Tensor(stride),bias,weight)
        for j in range(batch_size):
            
            for q in range(out_channel):
                n =0
                for i in range(0,input_size-kernel_size+1,stride):#0,output_size,stride):
                    inn = i
                    outt= i+kernel_size
                    segment = x.data[j,:,inn:outt].flatten()
                    w = weight.data[q,:,:].flatten()
                    z = np.inner(w,segment) 
                    out[j,q,n] = z + bias.data[q]
                    n = n+1      
        requires_grad = x.requires_grad
        return tensor.Tensor(out, requires_grad=requires_grad,is_leaf=not requires_grad)#, is_parameter = True)
        # TODO: Put output into tensor with correct settings and return 
        #raise NotImplementedError("Implement functional.Conv1d.forward()!")
    
    @staticmethod
    def backward(ctx, grad_output):
        #print(type(grad_output))
        x,stride,bias,weight = ctx.saved_tensors
        stride = stride.data
    
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        
        w_dx = np.zeros(weight.shape) 
        x_dx = np.zeros(x.shape)
        bias_dx = np.zeros(bias.shape)
        
        for j in range(batch_size):
            for q in range(out_channel):
                n =0
                for i in range(0,input_size-kernel_size+1,stride):
                    inn = i
                    outt= i+kernel_size
                    w_dx[q,:,:] += x.data[j,:,inn:outt]*grad_output.data[j,q,n]
                    x_dx[j,:,inn:outt] += weight.data[q,:,:]*grad_output.data[j,q,n]
                    bias_dx[q]  += grad_output.data[j,q,n]#*np.ones(bias.data[q].shape)
                    n = n+1
                    
        return tensor.Tensor(x_dx), tensor.Tensor(w_dx) , tensor.Tensor(bias_dx)      
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        #return tensor.Tensor(grad)
        #raise NotImplementedError("Implement functional.Conv1d.backward()!")

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensor.Tensor(grad)
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensor.Tensor(grad)

#..................................................next phase ..........................................

class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        ctx.save_for_backward(x, tensor.Tensor(indices))
        """print(x)
        print("...............................................")"""
        #raise NotImplementedError('Implemented Slice.forward')
        sliced_x = x.data[indices]
        requires_grad= x.requires_grad
        return tensor.Tensor(sliced_x,requires_grad=requires_grad,is_leaf=not requires_grad)

    @staticmethod
    def backward(ctx,grad_output):
        #raise NotImplementedError('Implemented Slice.backward')
        x,indices = ctx.saved_tensors
        indices = indices.data
        indices = tuple(indices.reshape(1,-1)[0])
     
        grad_out = np.zeros((x.shape))
        grad_out[indices] = np.ones(x.data[indices].shape)*grad_output.data
      
        return tensor.Tensor(grad_out)

class Cat(Function):
    @staticmethod
    def forward(ctx,dim,*seq):
        '''
        Args:
            dim (int): The dimension along which we concatenate our tensors
            seq (list of tensors): list of tensors we wish to concatenate
        '''
        #raise NotImplementedError('Implement Cat.forward')
        #print(seq)
        ctx.save_for_backward(*seq,tensor.Tensor(dim))
        x_cat = seq[0].data
        for i in range(1,len(seq)):
            x_cat = np.concatenate((x_cat,seq[i].data),axis =dim) #figure out some forloop or something

        requires_grad= seq[0].requires_grad
        return tensor.Tensor(x_cat,requires_grad=requires_grad,is_leaf=not requires_grad)

    @staticmethod
    def backward(ctx,grad_output):
        # NOTE: Be careful about handling the None corresponding to dim that you return, in case your implementation expects that for non-tensors
        #raise NotImplementedError('Implement Cat.backward')
        *seq,dim = ctx.saved_tensors
        output = []
        store = []
        inn =0
        outt = 0
        dim = dim.data
        grad_out = grad_output.data
        for i in seq:
                outt = inn + i.shape[dim]
                store.append(outt)
                inn = outt
                
        output = np.split(grad_out,store[0:-1],dim)
        
        return (*output,None)
            
class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, a,dim):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        
        
        x = list(a.shape)
        x.insert(dim,1)
        shape= tuple(x)
        #ctx.shape,Tensor(shape) = a.shape
        #self.data = self.data.reshape(s)
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        ctx.shape = c.shape
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.reshape(ctx.shape)), None