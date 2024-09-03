import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os
import sys

# Añadir el directorio raíz del proyecto al path de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train_model import train_model, evaluate_model
from src.models.predict_model import prepare_input, predict

class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Crear un DataFrame de prueba
        cls.test_data = pd.DataFrame({
            'Age': [30, 40, 50, 60, 70],
            'Flight Distance': [1000, 2000, 3000, 4000, 5000],
            'Inflight wifi service': [3, 4, 5, 2, 1],
            'Departure/Arrival time convenient': [4, 3, 5, 1, 2],
            'Ease of Online booking': [3, 5, 4, 2, 1],
            'Gate location': [2, 3, 4, 5, 1],
            'Food and drink': [4, 5, 3, 2, 1],
            'Seat comfort': [5, 4, 3, 2, 1],
            'Inflight entertainment': [3, 4, 5, 2, 1],
            'On-board service': [4, 5, 3, 2, 1],
            'Leg room service': [5, 4, 3, 2, 1],
            'Baggage handling': [3, 4, 5, 2, 1],
            'Checkin service': [4, 5, 3, 2, 1],
            'Inflight service': [5, 4, 3, 2, 1],
            'Cleanliness': [3, 4, 5, 2, 1],
            'Departure Delay in Minutes': [10, 20, 30, 40, 50],
            'Arrival Delay in Minutes': [15, 25, 35, 45, 55],
            'satisfaction': [1, 0, 1, 0, 1]
        })

    @patch('src.models.train_model.RandomForestClassifier')
    def test_train_model(self, mock_rf):
        mock_model = MagicMock()
        mock_rf.return_value = mock_model

        X = self.test_data.drop('satisfaction', axis=1)
        y = self.test_data['satisfaction']

        model = train_model(X, y)

        mock_rf.assert_called_once()
        mock_model.fit.assert_called_once_with(X, y)
        self.assertEqual(model, mock_model)

    def test_evaluate_model(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 0, 1, 0, 1]

        X = self.test_data.drop('satisfaction', axis=1)
        y = self.test_data['satisfaction']

        accuracy, report, conf_matrix = evaluate_model(mock_model, X, y)

        self.assertEqual(accuracy, 1.0)
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1-score', report)
        self.assertEqual(conf_matrix.shape, (2, 2))

    def test_prepare_input(self):
        input_data = {
            'Age': 30,
            'Flight Distance': 1000,
            'Inflight wifi service': 3,
            'Departure/Arrival time convenient': 4,
            'Ease of Online booking': 3,
            'Gate location': 2,
            'Food and drink': 4,
            'Seat comfort': 5,
            'Inflight entertainment': 3,
            'On-board service': 4,
            'Leg room service': 5,
            'Baggage handling': 3,
            'Checkin service': 4,
            'Inflight service': 5,
            'Cleanliness': 3,
            'Departure Delay in Minutes': 10,
            'Arrival Delay in Minutes': 15
        }

        prepared_input = prepare_input(input_data)

        self.assertIsInstance(prepared_input, pd.DataFrame)
        self.assertEqual(prepared_input.shape[1], 18)  # 18 features expected
        self.assertEqual(prepared_input.shape[0], 1)   # 1 row expected

    @patch('src.models.predict_model.joblib.load')
    def test_predict(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_load.return_value = mock_model

        input_data = pd.DataFrame({
            'Age': [30],
            'Flight Distance': [1000],
            'Inflight wifi service': [3],
            'Departure/Arrival time convenient': [4],
            'Ease of Online booking': [3],
            'Gate location': [2],
            'Food and drink': [4],
            'Seat comfort': [5],
            'Inflight entertainment': [3],
            'On-board service': [4],
            'Leg room service': [5],
            'Baggage handling': [3],
            'Checkin service': [4],
            'Inflight service': [5],
            'Cleanliness': [3],
            'Departure Delay in Minutes': [10],
            'Arrival Delay in Minutes': [15]
        })

        prediction, probability = predict(mock_model, input_data)

        self.assertEqual(prediction[0], 1)
        self.assertAlmostEqual(probability[0][1], 0.7)

if __name__ == '__main__':
    unittest.main()