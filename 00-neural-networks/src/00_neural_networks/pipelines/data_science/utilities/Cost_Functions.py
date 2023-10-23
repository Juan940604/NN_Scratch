# Standard libraries
import numpy as np
import random
import sys


# Define sigmoid and sigmoid_prime function (derivative of sigmoid with respect to z)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

### Define the quadratic and cross entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a,y):
        """ Return the cost associated with an output ``a`` and desired output ``y``
            ``a`` and ``y`` is an array with dimension 10 
            """
        return (0.5*np.linalg.norm(a-y)**2)
    
    @staticmethod
    def delta(z,a,y):
        """ Return the error delta form the output layer. Derivate C with respect to z in the last layer """
        return (a-y)*sigmoid_prime(z)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a,y):
        """ Return the cost associated with an output ``a`` and desired output ``y``. 
            Note that np.nan_to_num is used to ensure numerical stability. 
            In particular, if both ``a`` and ``y`` hace a 1.0 in the same slot, then
            the expression  (1-y)*np.log(1-a) returns na. The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """

        return np.sum(np.nan_to_num(-y*np.log(a) -(1-y)*np.log(1-a) ))
    
    @staticmethod
    def delta(z,a,y):
         """ Return the error delta form the output layer. Derivate C with respect to z in the last layer.
             Note that the parameter ``z`` is not used by the method. It is included in the method's parameters in 
             order to make the interface consistent with the delta method for other cost classes
            """
         return (a-y)
    
