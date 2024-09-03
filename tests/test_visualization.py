import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Añadir el directorio raíz del proyecto al path de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.visualize import plot_feature_importance, plot_confusion_matrix

class TestVisualization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Crear un DataFrame de prueba
        cls.test_data = pd.DataFrame({
            'Age': [30, 40, 50, 60, 70],
            'Flight Distance': [1000, 2000, 3000, 4000, 5000],
            'Inflight wifi service': [3, 4, 5, 2, 1],
            'satisfaction': [1, 0, 1, 0, 1]
        })
        
        # Crear un modelo mock
        cls.mock_model = MagicMock()
        cls.mock_model.feature_importances_ = np.array([0.3, 0.2, 0.5])

    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance(self, mock_savefig):
        # Llamar a la función de visualización
        plot_feature_importance(self.mock_model, self.test_data.drop('satisfaction', axis=1))
        
        # Verificar que se llamó a savefig
        mock_savefig.assert_called_once_with('reports/figures/feature_importance.png')
        
        # Verificar que se creó el directorio si no existía
        self.assertTrue(os.path.exists('reports/figures'))

    @patch('matplotlib.pyplot.savefig')
    def test_plot_confusion_matrix(self, mock_savefig):
        # Crear datos de prueba para la matriz de confusión
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        # Llamar a la función de visualización
        plot_confusion_matrix(y_true, y_pred)
        
        # Verificar que se llamó a savefig
        mock_savefig.assert_called_once_with('reports/figures/confusion_matrix.png')
        
        # Verificar que se creó el directorio si no existía
        self.assertTrue(os.path.exists('reports/figures'))

    def test_plot_feature_importance_invalid_input(self):
        # Probar con un modelo que no tiene feature_importances_
        invalid_model = MagicMock()
        invalid_model.feature_importances_ = None
        
        with self.assertRaises(AttributeError):
            plot_feature_importance(invalid_model, self.test_data.drop('satisfaction', axis=1))

    def test_plot_confusion_matrix_invalid_input(self):
        # Probar con arrays de diferentes tamaños
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 1, 1])
        
        with self.assertRaises(ValueError):
            plot_confusion_matrix(y_true, y_pred)

if __name__ == '__main__':
    unittest.main()