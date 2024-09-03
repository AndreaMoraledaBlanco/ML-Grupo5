import unittest
import pandas as pd
import os
import sys

# Añadir el directorio raíz del proyecto al path de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.build_features import create_total_delay, create_service_rating

class TestFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Crear un DataFrame de prueba
        cls.test_data = pd.DataFrame({
            'Departure Delay in Minutes': [10, 20, 30, 40, 50],
            'Arrival Delay in Minutes': [15, 25, 35, 45, 55],
            'Inflight wifi service': [3, 4, 5, 2, 1],
            'Food and drink': [4, 5, 3, 2, 1],
            'Seat comfort': [5, 4, 3, 2, 1],
            'Inflight entertainment': [3, 4, 5, 2, 1],
            'On-board service': [4, 5, 3, 2, 1],
            'Leg room service': [5, 4, 3, 2, 1],
            'Baggage handling': [3, 4, 5, 2, 1],
            'Checkin service': [4, 5, 3, 2, 1],
            'Inflight service': [5, 4, 3, 2, 1],
            'Cleanliness': [3, 4, 5, 2, 1]
        })

    def test_create_total_delay(self):
        df = self.test_data.copy()
        df_with_total_delay = create_total_delay(df)

        self.assertIn('total_delay', df_with_total_delay.columns)
        self.assertEqual(
            df_with_total_delay['total_delay'].iloc[0],
            df['Departure Delay in Minutes'].iloc[0] + df['Arrival Delay in Minutes'].iloc[0]
        )

    def test_create_service_rating(self):
        df = self.test_data.copy()
        df_with_service_rating = create_service_rating(df)

        self.assertIn('average_service_rating', df_with_service_rating.columns)
        service_columns = ['Inflight wifi service', 'Food and drink', 'Seat comfort', 'Inflight entertainment', 
                           'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 
                           'Inflight service', 'Cleanliness']
        expected_rating = df[service_columns].iloc[0].mean()
        self.assertAlmostEqual(df_with_service_rating['average_service_rating'].iloc[0], expected_rating)

if __name__ == '__main__':
    unittest.main()