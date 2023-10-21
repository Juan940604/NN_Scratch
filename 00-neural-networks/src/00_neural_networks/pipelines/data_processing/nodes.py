"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.7
"""

import numpy as np
import importlib


def vectorized_result(j):
    """ Return a 10-dimensional unit vector with a 1.0 in the jth position
        and zeros elsewhere. This is used to convert a digit (0,...,9) into 
        a corresponding desired output from the neural network
    """
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def split_data(mnist: tuple) ->  tuple:
    tr_d, va_d, te_d = mnist
    training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0]] # Transform the shape from (784,) to (784,1)
    training_results = [vectorized_result(x) for x in tr_d[1]] # Transform the output into vector with 1.0 in the output result
    training_data = list(zip(training_inputs,training_results))
    validation_inputs = [np.reshape(x, (784,1)) for x in va_d[0]] # Transform the shape from (784,) to (784,1)
    validation_results = [vectorized_result(x) for x in va_d[1]] # Transform the output into vector with 1.0 in the output result
    validation_data = list(zip(validation_inputs,validation_results))
    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (tuple([training_data,validation_data,test_data]))
