"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import greet


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=greet,    
            inputs="model_input_table",
            outputs=None ,     
            name="gret",
        ),
    ])
