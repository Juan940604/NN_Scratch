"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=["model_input_table",'params:model_options'],
            outputs='nn_trained',
            name='train_neural_network',
            ),
    ])
