import unittest
from flask_testing import TestCase
from backend import app, db
import json
import os


class Test_API(TestCase):

    def create_app(self):
        app.config['TESTING'] = True
        return app

    # test the /predict endpoint
    def test_predict_diamond_value(self):
        response = self.client.post('/predict', json={
            'carat': 0.23,
            'cut': 'Ideal',
            'color': 'E',
            'clarity': 'VS2',
            'depth': 61.5,
            'table': 55,
            'x': 3.95,
            'y': 3.98,
            'z': 2.43
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('predicted_price', response.json)

    # test the /predict endpoint with missing data
    def test_predict_diamond_value_missing_data(self):
        response = self.client.post('/predict', json={
            'carat': 0.23,
            'cut': 'Ideal',
            'color': 'E'
            # missing fields
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

    # test the wrong datatype for a field in the /predict endpoint
    def test_predict_diamond_value_wrong_datatype(self):
        response = self.client.post('/predict', json={
            'carat': 0.23,
            'cut': 'Ideal',
            'color': 'E',
            'clarity': 'VS2',
            'depth': 61.5,
            'table': 55,
            'x': 3.95,
            'y': 3.98,
            'z': '2.43'
            # wrong type for 'z'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

    # test the /similar-diamonds endpoint
    def test_similar_diamonds(self):
        response = self.client.post('/similar-diamonds', json={
            'cut': 'Ideal',
            'color': 'E',
            'clarity': 'VS2',
            'carat': 0.23,
            'n': 5
        })
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertIsInstance(json_data, list)

    # test the /similar-diamonds endpoint with missing data
    def test_similar_diamonds_missing_data(self):
        response = self.client.post('/similar-diamonds', json={
            'cut': 'Ideal',
            'color': 'E'
            # missing fields
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

    # test the wrong datatype for a field in the /similar-diamonds endpoint
    def test_similar_diamonds_wrong_datatype(self):
        response = self.client.post('/similar-diamonds', json={
            'cut': 'Ideal',
            'color': 'E',
            'clarity': 'VS2',
            'carat': 0.23,
            'n': '5'
            # wrong type for 'n'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)


if __name__ == '__main__':
    unittest.main()
