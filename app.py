from flask import Flask, request, jsonify
import pickle
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Manual CORS — no library needed
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_features.json', 'r') as f:
    features = json.load(f)

print("Model loaded. Features:", features)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'Car Price API is running'})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    try:
        data = request.json
        required = ['curb_weight', 'engine_size',
                    'horsepower', 'highway_mpg', 'width']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        sample = pd.DataFrame([[
            data['curb_weight'],
            data['engine_size'],
            data['horsepower'],
            data['highway_mpg'],
            data['width']
        ]], columns=features)

        sample_scaled = scaler.transform(sample)
        price = model.predict(sample_scaled)[0]

        return jsonify({
            'predicted_price': round(float(price), 2),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
