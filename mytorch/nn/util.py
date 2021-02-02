from mytorch import tensor
import numpy as np

class PackedSequence:
    
    '''
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples
    
    ATTENTION: The "argument batch_size" in this function should not be confused with the number of samples in the batch for which the PackedSequence is being constructed. PLEASE read the description carefully to avoid confusion. The choice of naming convention is to align it to what you will find in PyTorch. 

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features ) 
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,) - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    '''
    def __init__(self,data,sorted_indices,batch_sizes):
        
        # Packed Tensor
        self.data = data # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices # Sorted Indices
        
        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes # Batch sizes
    
    def __iter__(self):
        yield from [self.data,self.sorted_indices,self.batch_sizes]
    
    def __str__(self,):
        return 'PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})'.format(str(self.data),str(self.sorted_indices),str(self.batch_sizes))


def pack_sequence(sequence): 
    '''
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K) where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape ( total number of timesteps (sum) across all samples in the batch, # features )
    '''
    #print(sequence)
    
    # TODO: INSTRUCTIONS
    # Find the sorted indices based on number of time steps in each sample
    # Extract slices from each sample and properly order them for the construction of the packed tensor. __getitem__ you defined for Tensor class will come in handy
    # Use the tensor.cat function to create a single tensor from the re-ordered segements
    # Finally construct the PackedSequence object
    # REMEMBER: All operations here should be able to construct a valid autograd graph.

    #raise NotImplementedError('Implement pack_Sequence!')
    sorted_indices = []
    stored_tensor =[]
    store = []
    batch_size = []
    batch_number = 0
    for x,seq in enumerate(sequence):
        sorted_indices.append([seq.shape[0],x,seq])
    #print(sorted_indices)
    sorted_indices = sorted(sorted_indices,reverse= True)
    sorted_indices1 = np.array([i[1] for i in sorted_indices])
    stored_tensor = [i[2] for i in sorted_indices]
    #print(stored_tensor)
    packed_tensor =[[] for i in range(stored_tensor[0].shape[0])]
    
    for c,q in enumerate(stored_tensor):
      for d,x in enumerate(q):
        packed_tensor[d].append(q[d])
    batch_size1= np.array([len(i) for i in packed_tensor])
    
    for j in range(stored_tensor[0].shape[0]):
      batch_number = 0
      for c,i in enumerate(stored_tensor):
        if (j < i.shape[0]):
          input1= i[j]
          batch_number = batch_number +1
          store.append(input1.unsqueeze(dim=0))
      batch_size.append(batch_number)
    batch_size = np.array(batch_size)

    packed_tensor = tensor.cat(store,dim=0)
 
            
    return PackedSequence(packed_tensor,sorted_indices1, batch_size1)


        

def unpack_sequence(ps):
    '''
    Given a PackedSequence, this unpacks this into the original list of tensors.
    
    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    '''
    #print(ps.data)
    # TODO: INSTRUCTIONS
    # This operation is just the reverse operation of pack_sequences
    # Use the ps.batch_size to determine number of time steps in each tensor of the original list (assuming the tensors were sorted in a descending fashion based on number of timesteps)
    # Construct these individual tensors using tensor.cat
    # Re-arrange this list of tensor based on ps.sorted_indices
    outt =0
    #print(ps.data)
    inn =0
    unpack = []
    unpacked =[]
    full_unpacked = []
    sorted_full_unpacked =[]
    data = ps.data
    
    for count,i in enumerate(ps.batch_sizes):
        outt = inn+i
        new1 = data[inn:outt,:]
        unpack.append(new1)
        inn = outt
    
    for j in range(len(ps.sorted_indices)):
        unpacked = []
        for count, i in enumerate(unpack):
            if(j < len(i)):
                input1 = i[j,:]
                unpacked.append(input1.unsqueeze(dim=0))
        full_unpacked1 = tensor.cat(unpacked,dim=0)
        full_unpacked.append(full_unpacked1)

    
    full_unpacked_trial = np.array(full_unpacked)
    #find a way to sort
    
    
    """count =0
    while True:
        if(count==len(ps.sorted_indices)):
            break;
        for p,i in enumerate(ps.sorted_indices):
            if(count == i):
                sorted_full_unpacked.append(full_unpacked[p])
                count = count+1"""
        
    sorted_full_unpacked = full_unpacked_trial[ps.sorted_indices]
    #print(type(ps.sorted_indices),ps.sorted_indices)
    return sorted_full_unpacked#[ps.sorted_indices] 

