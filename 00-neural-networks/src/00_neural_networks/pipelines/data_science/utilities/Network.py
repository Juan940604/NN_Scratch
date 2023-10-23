## Main Libraries
import numpy as np 
import jason
import sys
from  .Cost_Functions import QuadraticCost, CrossEntropyCost, sigmoid, sigmoid_prime
from .early_stopping import early_stopping

## Main Network Class

class Network(object):

    def __init__(self, sizes: list, cost = CrossEntropyCost, early_stopping = early_stopping):
        """
        -  sizes: It is a list that contains the number of neurons in each neuron from the input layer to the output layer
        -  cost:  It is a function that defines the cost function. CrossEntropyCost is selected by default
        -  early_stopping: It is a function that us used to interrupt the training when we are satisfied with the accruracy of the method    
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        self.early_stopping = early_stopping

    def default_weight_initializer(self):
        """
        This function initializes each weight using a Gaussian distribution with mean 0 and standar deviation 1 
        over the square root of the number of weights connecting to the same neuron (1/K)
        Initialize the biases using a Gaussian distribution with mean 0 and standar deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """

        self.biases = [np.random.randn(y,1) for y in self.sizes[1:] ]
        self.weights = [np.random.rand(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def large_weight_initializer(self):
        """
        This function initializes each weight using a Gaussian distribution with mean 0 and standar deviation 1 
        Initialize the biases using a Gaussian distribution with mean 0 and standar deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.rand(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def feedforward(self,a):
        """ 
            The feedforwar returns the predictions of the neural networks
            Return the output of the network if ``a`` is the input. 
            Input 
            - a: the neurons of the input layer 
            
            Return 
            - a: the neurons of the output layer

        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a 

    def SGD(self,
            training_data,
            epochs,
            mini_bathc_size,
            eta,
            regularization,
            lmbda = 0.0, 
            nn = 10,
            epsilon= 0.1,
            evaluation_data = None,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False 
            ):
        """ 
            This function is the optimizer. It helps us to provide the new guess of parameters for each iteration through Stochastic Gradient Descent (SGD)
            - training_data: It is a list of tuples ``(x,y)`` representing the training inputs and the desired outputs
            - epochs: The number of epochs used to train ( An epoch is simply a cycle in which all the samples (grouped in batches) are used to train the model)
            - mini_batch_size: It is an integer number that indicates tha size of each batch
            - eta: This is the learning rate
            - regularization: It is a the type of regularization ``L1`` or ``L2``
            - lmbda: It is the  regularization parameter
            - nn: It is the number of epochs needed to apply the early_stopping method
            - epsilon: It is the value used as a threshold criterion for convergence
            - evaluation_data: It is a list of tuples ``(x,y)`` representing the validation inputs and the desired validation outputs

            We can monitor the cost and accuracy on either the evaluation data or the training data. 
            The method returns a tuple containing four lists: 
             1. the (per-epoch) costs on the evaluation data, 
             2. the accuracies on the evaluation data, 
             3. the costs on the training data,
             4. the accuracies on the training data.
             
            All values are evaluated at the end of each training epoch
        """

        if evaluation_data: n_data = len(evaluation_data)

        n = len(training_data)





