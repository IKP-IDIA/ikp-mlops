from mlflow.models import validate_serving_input
import pandas as pd
INPUT_EXAMPLE =X_test.iloc[[0]]

model_uri = 'runs:/8c7c175a3c764f4aaf48677c5666d25a/model'

from mlflow.models import convert_input_example_to_serving_input

serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)

validate_serving_input(model_uri, serving_payload)