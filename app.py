from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import pandas as pd
import numpy as np
import math

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoder
model = tf.keras.models.load_model('DentNNv2.h5')


class Item:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@app.post('/predict')
async def predict(request: Request):
    try:
        input_data = await request.json()

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

        return {"prediction": rounded_prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.options('/predict')
async def handle_options():
    return {"message": "Options"}
