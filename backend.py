from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from utils import get_latest_model, preprocessing_xgb, get_similar_diamonds

app = Flask(__name__)

# set the database URI to the value of the DATABASE_URI environment variable (test or production db)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URI', 'sqlite:///requests_responses.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class RequestResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    endpoint = db.Column(db.Text)
    request_data = db.Column(db.Text)
    response_data = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, endpoint, request_data, response_data):
        self.endpoint = endpoint
        self.request_data = request_data
        self.response_data = response_data


with app.app_context():
    db.create_all()


model_xgb = get_latest_model('xgb')

diamonds = pd.read_csv('data/diamonds.csv')

# function for saving request and response data to the database


def save_request_response(endpoint, request_data, response_data):
    request_response = RequestResponse(
        endpoint=endpoint,
        request_data=str(request_data),
        response_data=str(response_data)
    )
    db.session.add(request_response)
    db.session.commit()


# predict the price of a diamond with xgb given its features

@app.route('/predict', methods=['POST'])
def predict_diamond_value():
    data = request.json

    valid, error = validate_predict_diamond_value(data)
    if not valid:
        return jsonify({'error': error}), 400

    carat = data['carat']
    cut = data['cut']
    color = data['color']
    clarity = data['clarity']
    depth = data['depth']
    table = data['table']
    x = data['x']
    y = data['y']
    z = data['z']

    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z],
    })

    input_data = preprocessing_xgb(input_data)
    predicted_price = model_xgb.predict(input_data)[0]

    response = {'predicted_price': f"{int(predicted_price)}$"}

    save_request_response(f'/predict', data, response)

    return jsonify(response)

# given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.


@app.route('/similar-diamonds', methods=['POST'])
def similar_diamonds():
    data = request.json

    valid, error = validate_similar_diamonds(data)
    if not valid:
        return jsonify({'error': error}), 400

    cut = data['cut']
    color = data['color']
    clarity = data['clarity']
    carat = data['carat']
    n = data['n']

    similar_diamonds = get_similar_diamonds(
        diamonds, cut, color, clarity, carat, n)

    if similar_diamonds.empty:
        response = jsonify({'error': 'No similar diamonds found'})

    else:
        response = similar_diamonds.to_json(orient='records')

    save_request_response('/similar-diamonds', data, response)

    return response

# validate the data type for the similar-diamonds endpoint


def validate_similar_diamonds(data):
    required_fields = {
        'n': int,
        'cut': str,
        'color': str,
        'clarity': str,
        'carat': (float, int)
    }

    for field, field_type in required_fields.items():
        if field not in data:
            return False, f"Missing data: {field}"
        if isinstance(field_type, tuple):
            if not isinstance(data[field], field_type):
                return False, f"Incorrect type for {field}. Expected {field_type[0].__name__}."
        else:
            if not isinstance(data[field], field_type):
                return False, f"Incorrect type for {field}. Expected {field_type.__name__}."

    return True, None

# validate the data type for the predict endpoint


def validate_predict_diamond_value(data):
    required_fields = {
        'carat': (float, int),
        'cut': str,
        'color': str,
        'clarity': str,
        'depth': (float, int),
        'table': (float, int),
        'x': (float, int),
        'y': (float, int),
        'z': (float, int)
    }

    for field, field_type in required_fields.items():
        if field not in data:
            return False, f"Missing data: {field}"
        if isinstance(field_type, tuple):
            if not isinstance(data[field], field_type):
                return False, f"Incorrect type for {field}. Expected {field_type[0].__name__}."
        else:
            if not isinstance(data[field], field_type):
                return False, f"Incorrect type for {field}. Expected {field_type.__name__}."

    return True, None


if __name__ == '__main__':
    app.run(debug=True)
