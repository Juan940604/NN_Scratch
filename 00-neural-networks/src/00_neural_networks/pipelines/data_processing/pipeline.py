"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs="mnist",
            outputs="model_input_table",
            name='split_data'
        ),
    ])



