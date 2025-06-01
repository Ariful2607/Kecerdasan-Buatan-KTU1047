from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
model = joblib.load(model_path)

# Mapping from numeric labels to species names
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    species = species_mapping[int(prediction[0])]
    return jsonify({'prediction': species})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
