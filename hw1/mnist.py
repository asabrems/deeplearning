"""Problem 3 - Training on MNIST"""
import numpy as np
from  mytorch.tensor import Tensor
#import mytorch.nn.functional as F
from mytorch.optim.sgd import SGD
from mytorch.nn.activations import ReLU
from mytorch.nn.loss import CrossEntropyLoss as XELoss
from mytorch.nn.linear import Linear
from mytorch.nn.sequential import Sequential
#from mytorch.nn.module import Module


# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    model = Sequential(Linear(784,20),ReLU(),Linear(20,10))
    optimizer = SGD(model.parameters(),lr= 0.1,momentum=0.0)
    criterion = XELoss()
    
    
    
    # TODO: Call training routine (make sure to write it below)
    val_accuracies = None
    val_accuracies = train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3)
    
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    
    val_accuracies = []
    for i in range(num_epochs):
        model.train()
        c = np.arange((train_x.shape[0]))
        np.random.shuffle(c)
        train_x = train_x[c]
        train_y = train_y[c]
        batch_x = np.array_split(train_x,BATCH_SIZE)
        batch_y = np.array_split(train_y,BATCH_SIZE)
        batches = zip(batch_x,batch_y)
        
        for i,(batch_data, batch_labels) in enumerate(batches):
            optimizer.zero_grad()
            out = model.forward(Tensor(batch_data))
            loss = criterion.forward(out, Tensor(batch_labels))
            loss.backward()
            optimizer.step()
            if i %100:
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                model.train()
    
    # TODO: Implement me! (Pseudocode on writeup)
    return val_accuracies

def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    #TODO: implement validation based on pseudocode
    
    model.eval()
    num_correct = 0
    batch_x = np.array_split(val_x,BATCH_SIZE)
    batch_y = np.array_split(val_y,BATCH_SIZE)
    batches = zip(batch_x,batch_y)
    
    for i,(batch_data, batch_labels) in enumerate(batches):
        out= model.forward(Tensor(batch_data))
        predict = np.argmax(out.data, axis=1)
        num_correct += (predict == batch_labels)
    accuracy = num_correct.sum() /len(val_x)
    return accuracy
