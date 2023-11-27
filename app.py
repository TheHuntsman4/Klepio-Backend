from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import numpy as np
import math

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes with default settings

# Load model and encoder
model = tf.keras.models.load_model('DentNNv2.h5')

class Item:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json

        # Convert string values to numeric types
        input_data = {key: float(value) for key, value in input_data.items()}

        # Create an Item instance
        input_item = Item(**input_data)

        # Convert Item object to DataFrame
        input_df = pd.DataFrame([input_item.__dict__])

        # Fill NaN values with 0
        input_df = input_df.fillna(0)

        # Extract inputs as a numpy array
        inputs = input_df.values

        # Make predictions using the pre-trained model
        prediction = model.predict(inputs).tolist()

        # Round down the predicted value to the nearest integer
        rounded_prediction = math.floor(prediction[0][0])

        return jsonify({"prediction": rounded_prediction})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    return jsonify({"message": "Options"})

if __name__ == '__main__':
    app.run(debug=True)
