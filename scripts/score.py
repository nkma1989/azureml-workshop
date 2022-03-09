import os
import joblib
import numpy as np
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
from inference_schema.schema_decorators import input_schema, output_schema


def init():
    """
    Loads artifacts used for ML predictions
    """
    global model

    # Defining model path
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    # deserialize the model file back into a model
    model = joblib.load(model_path)
    
# Sample schemas
standard_sample_input = {
                    'AGE': 41.0,
                    'SEX': 2.0,
                    'BMI': 32.0,
                    'BP': 109.0,
                    'S1': 251.0,
                    'S2': 170.6,
                    'S3': 49.0,
                    'S4': 5.0,
                    'S5': 5.0562,
                    'S6': 103.0
                    }
standard_sample_output = {"predictions": 1}


# Defining input and output schema
@input_schema("data", StandardPythonParameterType(standard_sample_input))
@output_schema(StandardPythonParameterType(standard_sample_output))
def run(data):
    """
    Predicts
    """
    try:
        # Converting dict payload to list
        list_values = list(data.values())
        # Converting to numpy array and reshaping to fit model input
        np_array = np.array(list_values).reshape(1, len(list_values))
        prediction = model.predict(np_array).tolist()
        return {"predictions": prediction}
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return {"error": result}