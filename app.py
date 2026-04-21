# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ── Load model and scaler on startup ──
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Model and scaler loaded successfully")

# ── Home route — test if API is running ──
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'Car Price API is running'})

# ── Predict route ──
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json

        # Validate all fields exist
        required = ['curb_weight', 'engine_size',
                    'horsepower', 'highway_mpg', 'width']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Build feature array
        # Order must match how scaler was trained
        features = np.array([[
            data['curb_weight'],
            data['engine_size'],
            data['horsepower'],
            data['highway_mpg'],
            data['width']
        ]])

        # Scale the input
        features_scaled = scaler.transform(features)

        # Predict
        price = model.predict(features_scaled)[0]

        return jsonify({
            'predicted_price': round(float(price), 2),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Run the app ──
if __name__ == '__main__':
    app.run(debug=True)