__author__ = 'kkboy'

import numpy as np


def gen_data(min_length=50, max_length=55, n_batch=5):

    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        #i changed this to a constant
        #length=55

        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/2-1), 1] = 1
        X[n, np.random.randint(length/3), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    #X -= X.reshape(-1, 2).mean(axis=0)
    #y -= y.mean()
    return (X,y)



x,y = gen_data(n_batch=1)
print x
print y