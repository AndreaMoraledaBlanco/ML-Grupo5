import unittest
import sys
import os

# Añadir el directorio raíz del proyecto al path de Python
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def run_tests():
    # Descubrir y cargar las pruebas automáticamente
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')

    # Ejecutar las pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Salir con un código de estado no cero si alguna prueba falló
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    print(f"Buscando pruebas en: {os.path.join(project_root, 'tests')}")
    run_tests()