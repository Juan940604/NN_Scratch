## Main Libraries
import numpy as np 
import random
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
    

    
    def backprop(self,x,y):
        """
        Return a tuple ``(nabla_b,nabla_w)`` representing the gradient for the cost function C_x corresponding to one single sample.
        Input:
            - x: The features for a single sample of the training data
            - y: The target for a single sample of the training data
        Output:
            - nabla_b: It is a layer-by-layer lists of numpy arrays similar to self.b. It contains the derivatives of C_x with respect to the biases
            - nabla_w: It is a layer-by-layer lists of numpy arrays similar to self.w. It contains the derivatives of C_x with respect to the biases
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # List to store all the activations, layer by layer
        zs = [] # list to store all z vectors, layer by layer
        # Forward step for each layer
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward step
        ## For the last layer L
        delta = self.cost.delta(z=zs[-1],a=activations[-1],y=y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose)

        for l in range(2,self.num_layers):
            # l starts in 2
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose,delta)*sp
            nabla_b[-l] =  delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose)
        return(nabla_b, nabla_w)
    

    
    def update_mini_batch(self, mini_batch,eta,lmbda, n, regularization):
        """
            Update the network's weights and biases by applying gradient descent using ``backpropagation`` to a single minibatch.
            - minibatch: It is a list of tuples ``(x,y)``
            - eta: It is the learning rate
            - lmbda: It is the regularization parameter
            - n: It is the total size of the training data set
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # This for loop calculates the sum of the derivatives of the cost function with respect to the interested parameters using the mini_batch training data
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        if regularization == 'L2':
            self.weights = [(1-eta*lmbda/n)*w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights,nabla_w) ]
        elif regularization == 'L1':
            self.weights = [ w - (eta*lmbda/n)*np.sign(w) - (eta/len(mini_batch))*nw for w, nw in zip(self.weights,nabla_w) ]

        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)] 

    def total_cost(self,data,lmbda, convert=False):
        """
        Return the total cost for the data set ``data``
        """




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
        evaluation_cost, evaluation_accuracy = [],[]
        training_cost, training_accuracy = [],[]
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:(k+mini_bathc_size)] for k in range(0,n,mini_bathc_size) ]
            for mini_batch in mini_bathc_size:
                self.update_mini_batch(mini_batch,eta, lmbda, len(training_data), regularization)
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.
        




