"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.7
"""
import logging
from typing import Dict, Tuple
#from .utilities.Network import Network
from .utilities import  Network

def train_model(model_input_table: Tuple,params: Dict) -> Dict:
    training_data, validation_data, test_data = model_input_table

    # Hyperparameters
    # Layers and neurons
    sizes = params['sizes']
    # cost function
    cost = params['cost']
    if hasattr(Network, cost):
        cost = getattr(Network, cost)
    else: 
        print(f"Network has no function named {cost}")
    # Learning rate
    eta = params['eta']
    epochs = params['epochs']
    mini_batch_size = params['mini_batch_size']
    monitor_evaluation_accuracy = params['monitor_evaluation_accuracy']
    regularization = params['regularization']

    # Neural Network
    net = Network.Network(sizes=sizes,cost = cost)

    #
    logger = logging.getLogger(__name__)
    net.SGD(training_data=training_data,epochs= epochs, mini_batch_size= mini_batch_size, eta= eta,evaluation_data=test_data, monitor_evaluation_accuracy=monitor_evaluation_accuracy,regularization=regularization)

    data = {
                "sizes" : net.sizes,
                "weights" : [w.tolist() for w in net.weights],
                "biases" : [b.tolist() for b in net.biases],
                "cost": str(net.cost.__name__)
             }

    return data
    

