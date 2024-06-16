from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from utils import get_latest_model, preprocessing_xgb

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///requests_responses.db'
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

data = pd.read_csv('data/diamonds.csv')

# function for saving request and response data to the database


def save_request_response(endpoint, request_data, response_data):
    request_response = RequestResponse(
        endpoint=endpoint,
        request_data=str(request_data),
        response_data=str(response_data)
    )
    db.session.add(request_response)
    db.session.commit()


@app.route('/predict', methods=['POST'])
def predict_diamond_value():
    data = request.json
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

    input_data = preprocessing_xgb(input_data, predict=True)
    predicted_price = model_xgb.predict(input_data)[0]

    response = {'predicted_price': str(predicted_price)}

    save_request_response(f'/predict', data, response)
    return jsonify(response)

#  TODO: COMPLETE THIS


@app.route('/similar-diamonds', methods=['POST'])
def get_similar_diamonds():
    data = request.json
    cut = data['cut']
    color = data['color']
    clarity = data['clarity']
    carat = data['carat']

    similar_diamonds = data[
        (data['cut'] == cut) &
        (data['color'] == color) &
        (data['clarity'] == clarity) &
        (data['carat'].between(carat - 0.1, carat + 0.1))
    ].sample(n=5)

    similar_diamonds_json = similar_diamonds.to_json(orient='records')

    response = similar_diamonds_json
    save_request_response('/similar-diamonds', data, response)
    return response


if __name__ == '__main__':
    app.run(debug=True)
