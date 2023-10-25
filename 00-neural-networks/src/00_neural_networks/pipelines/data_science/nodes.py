"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.7
"""
import logging
from typing import Dict, Tuple

from .utilities import Network


def train_model(model_input_table: Tuple,parameter: Dict) -> Dict:
    training_data_list, validation_data_list, test_data_list = model_input_table
    

