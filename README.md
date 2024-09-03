# ML-Grupo5



Proyecto de Predicción de Satisfacción de Clientes de Aerolíneas
Este proyecto utiliza técnicas de aprendizaje supervisado para predecir la satisfacción de los clientes de una aerolínea basándose en diversas características de su experiencia de vuelo.

Estructura del Proyecto

ML-Grupo5/
│
├── data/
│   ├── raw/
│   │   └── airline_satisfaction.csv
│   └── processed/
│       ├── clean_airline_satisfaction.csv
│       └── featured_airline_satisfaction.csv
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   │   ├── create_sample_data.py
│   │   └── analyze_data.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       ├── __init__.py
│       ├── visualize.py
│       └── explore_data.py
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── models/
│   └── best_model.joblib
│
├── reports/
│   └── figures/
│       ├── feature_importance.png
│       ├── learning_curve.png
│       ├── confusion_matrix.png
│       └── roc_curve.png
│
├── tests/
│   └── test_prediction.py
│
├── venv/
│
├── .gitignore
├── requirements.txt
├── setup.py
├── README.md
└── Dockerfile
Instalación

Clone este repositorio:
Copygit clone https://github.com/tu-usuario/airline_satisfaction.git
cd airline_satisfaction

Cree un entorno virtual y actívelo:
Copypython -m venv venv
source venv/bin/activate  # En Windows use `venv\Scripts\activate`

Instale las dependencias:
Copypip install -r requirements.txt

Instale el proyecto en modo editable:
Copypip install -e .


Uso

Crear datos de muestra (si no se tienen datos reales):
Copypython src/data/create_sample_data.py

Preprocesamiento de datos:
Copypython src/data/preprocess.py

Análisis de datos:
Copypython src/data/analyze_data.py

Construcción de características:
Copypython src/features/build_features.py

Visualización y exploración de datos:
Copypython src/visualization/explore_data.py

Entrenamiento del modelo:
Copypython src/models/train_model.py

Hacer predicciones:
Copypython src/models/predict_model.py

Ejecutar pruebas:
Copypython -m unittest discover tests

Ejecutar la aplicación Streamlit:
Copystreamlit run app/main.py


Características del Proyecto

Generación de datos de muestra
Análisis exploratorio de datos
Preprocesamiento y limpieza de datos
Ingeniería de características
Entrenamiento de modelos de machine learning (Random Forest, Ensemble)
Optimización de hiperparámetros
Evaluación del modelo (precisión, matriz de confusión, curva ROC)
Visualizaciones detalladas del rendimiento del modelo
Pruebas unitarias
Interfaz de usuario con Streamlit para predicciones interactivas
Containerización con Docker

Docker
Para construir y ejecutar el proyecto usando Docker:

Construya la imagen:
Copydocker build -t airline_satisfaction .

Ejecute el contenedor:
Copydocker run -p 8501:8501 airline_satisfaction

Acceda a la aplicación Streamlit en su navegador en http://localhost:8501.

Desarrollo
Este proyecto utiliza setup.py para facilitar su instalación y desarrollo. Para instalar el proyecto en modo de desarrollo, ejecute:
Copypip install -e .
Esto permitirá que los cambios en el código se reflejen inmediatamente sin necesidad de reinstalar el paquete.
Dependencias Principales

pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib

Para una lista completa de dependencias, consulte requirements.txt.
Contribuir
Si desea contribuir a este proyecto, por favor:

Haga un Fork del repositorio
Cree una nueva rama (git checkout -b feature/AmazingFeature)
Haga commit de sus cambios (git commit -m 'Add some AmazingFeature')
Push a la rama (git push origin feature/AmazingFeature)
Abra un Pull Request

Licencia
Distribuido bajo la licencia MIT. Vea LICENSE para más información.
Contacto
Tu Nombre - tu@email.com
Link del proyecto: https://github.com/tu-usuario/airline_satisfaction