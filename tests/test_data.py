import unittest
import pandas as pd
import numpy as np
import os
import sys

# Añadir el directorio raíz del proyecto al path de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import preprocess_data, handle_missing_values, encode_categorical_variables, scale_numeric_features

class TestData(unittest.TestCase):

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
            'satisfaction': ['satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied']
        })

    def test_handle_missing_values(self):
        df = self.test_data.copy()
        df.loc[0, 'Age'] = np.nan
        df.loc[1, 'Flight Distance'] = np.nan
        df.loc[2, 'satisfaction'] = np.nan

        df_imputed = handle_missing_values(df)

        self.assertFalse(df_imputed.isnull().any().any())
        self.assertNotEqual(df_imputed.loc[0, 'Age'], np.nan)
        self.assertNotEqual(df_imputed.loc[1, 'Flight Distance'], np.nan)
        self.assertNotEqual(df_imputed.loc[2, 'satisfaction'], np.nan)

    def test_encode_categorical_variables(self):
        df = self.test_data.copy()
        df_encoded = encode_categorical_variables(df)

        self.assertTrue(pd.api.types.is_integer_dtype(df_encoded['satisfaction']))
        self.assertEqual(set(df_encoded['satisfaction'].unique()), {0, 1})

    def test_scale_numeric_features(self):
        df = self.test_data.copy()
        df_scaled = scale_numeric_features(df)

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'satisfaction':
                self.assertAlmostEqual(df_scaled[col].mean(), 0, places=7)
                self.assertAlmostEqual(df_scaled[col].std(), 1, places=7)

    def test_preprocess_data(self):
        input_filepath = 'tests/test_data.csv'
        output_filepath = 'tests/test_processed_data.csv'
        
        # Guardar datos de prueba en un archivo CSV
        self.test_data.to_csv(input_filepath, index=False)
        
        preprocess_data(input_filepath, output_filepath)
        
        # Verificar que el archivo de salida existe
        self.assertTrue(os.path.exists(output_filepath))
        
        # Cargar y verificar los datos procesados
        processed_data = pd.read_csv(output_filepath)
        self.assertFalse(processed_data.isnull().any().any())
        self.assertTrue(pd.api.types.is_integer_dtype(processed_data['satisfaction']))
        
        # Limpiar archivos de prueba
        os.remove(input_filepath)
        os.remove(output_filepath)

if __name__ == '__main__':
    unittest.main()