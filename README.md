# Car Price Prediction API

A Flask REST API that predicts car prices using a trained 
Random Forest model.

## Model Details

- Algorithm: Random Forest Regressor
- R² Score: 0.9475
- MAE: $1,702
- Features: 5 input variables

## Input Features

| Feature | Description | Range |
|---|---|---|
| curb_weight | Car weight in lbs | 1488 – 4066 |
| engine_size | Engine size in cc | 61 – 326 |
| horsepower | Engine power in HP | 48 – 288 |
| highway_mpg | Highway fuel efficiency | 16 – 54 |
| width | Car width in inches | 60 – 72 |

## API Endpoints

### GET /
Returns API status

### POST /predict
Predicts car price

**Request:**
```json
{
    "curb_weight": 2500,
    "engine_size": 130,
    "horsepower": 100,
    "highway_mpg": 30,
    "width": 65
}
```

**Response:**
```json
{
    "predicted_price": 14250.50,
    "status": "success"
}
```

## Tech Stack

Python · Flask · Scikit-learn · Gunicorn · Render

## Live API

https://car-price-api.onrender.com
