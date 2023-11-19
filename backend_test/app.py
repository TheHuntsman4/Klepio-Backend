from flask import Flask, request, jsonify
from flask_cors import CORS;
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load model and encoder
model = pickle.load(open('DentAIv1.pkl','rb')) 
encoder = pickle.load(open('encoder.pkl','rb'))

@app.route('/predict', methods=['POST'])
def predict():
    # Get data
    columns=['Chief complaint', 'Nature of Pain', 'Severity of pain', 'Onset  and mode of pain', 'Factors which worsens the pain', 'Is the swelling painful?', 'Has the swelling changed since it was first noticed? If yes how quickly?', 'Does the swelling changes during normal activities such as eating, speaking, etc?', 'Is the ulcer painful', 'Is there bleeding from the ulcer', 'Is there discharge from the ulcer?', 'Is there a foul smell from the ulcer?', 'Do the ulcers interfere with daily activities', 'Has the ulcer changed since first noticed?', 'Have you had similar ulcers?', 'Is there bleeding in the gums?', 'Is there pain in the gums', 'If any tooth/teeth is/are mobile, what is the degree of mobility']
    data = request.get_json() 
    data = {k: np.float32(v) for k, v in request.json.items()}
        
    df = pd.DataFrame(data, columns=columns, index=[0])
    
    df = df.apply(pd.to_numeric, errors='coerce')  

    # Fill NaN values with 0
    df = df.fillna(0)

    pred = model.predict(df)
    
    # Return result
    result = {'prediction': int(pred[0])}
    response=jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*') 
    return response

@app.route('/predict', methods=['OPTIONS'])  
def handle_options():
  return jsonify({})


if __name__ == '__main__':
    app.run(debug=True)
