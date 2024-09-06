import os
import gradio as gr
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import numpy as np

for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if f.endswith("airline_recoded.csv")]:
                os.chdir(dirpath)

airline_df = pd.read_csv('airline_recoded.csv')

X = airline_df.drop(columns = ['satisfaction'])
y = airline_df['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

xgb_model = joblib.load('../Modelos/xgboost_model.pkl')

y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

results = xgb_model.evals_result()
accuracy = accuracy_score(y_test, y_pred)

class_report = classification_report(y_test, y_pred)

cv_scores = cross_val_score(xgb_model, X, y, cv=5)

# Función para generar un gráfico
def generar_grafico_XGB(tipo_grafico):
    plt.figure(figsize=(8,6))
    if tipo_grafico == "Matriz de confusión":
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Neutral o No satisfecho", "Satisfecho"], yticklabels=["Neutral o No Satisfecho", "Satisfecho"])
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de confusión')
        
    elif tipo_grafico == "Curva ROC":
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
    elif tipo_grafico == "Overfitting":
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
        plt.legend(loc='upper right')
        plt.xlabel('Number of Trees')
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')
        return plt.gcf()
    
    plt.tight_layout()  # Ajustar el diseño
    return plt.gcf()


# Página 1
def pagina1():
    with gr.Row():
        gr.Markdown("# Modelo de ML decision tree con XGBoost")
    with gr.Row():
        gr.Markdown("""
                    ## Introducción

                    A continuación se presenta un modelo realizado con un algoritmo de arbol de decisión, construido utilizando XGBoost. 
                    

                    XGBoost es un algoritmo de _gradient boosting_ que ha ganado popularidad gracias a su velocidad y buenos resultados, especialmente para datos como los que estamos trabajando. 

                    XGBoost construye árboles de decisión de forma secuencial, donde cada árbol corrige errores generados en el árbol anterior. Incluye por defecto regularización (L1 y L2) para lidiar con problemas de overfitting, permitiendo además trabajar con otros hiperparámetros como la profundidad máxima (max_depth) o peso mínimo de nodos (min_child_weight).

                    Al trabajar de forma base con _gradient boosting_ es una forma muy sencilla de implementar técnicas de ensemble en un algoritmo de ML.

                    Para mejorar al máximo el rendimiento del modelo se han implementado los siguientes hiperparámetros:
                    - max_depth: Máxima profundidad del modelo ("ramas").
                    - gamma: Valor mínimo de pérdida para realizar una nueva partición del árbol. A mayor _gamma_ más conservador es el modelo.
                    - reg_alpha: Regularización L1, a mayor sea su valor más conservador es el modelo.
                    - min_child_weight: Suma mínima del peso en un "hijo". Si la partición del árbol resulta en una hoja nodo con una suma menor al valor especificado, el modelo dejará de realizar particiones.
                    - colsample_bytree: Submuestreo de columnas al construir el arbol. Especifica la fracción de columnas a submuestrear.
                    - eval_metric: La métrica para evaluar el modelo. Puede utilizarse varias, aquí usamos logloss, referida al valor negativo del log-likelihood.

                    Para determinar el mejor valor de los hiperparámetros se ha utilizado lo que se conoce como algoritmo de búsqueda ingenuo. Partimos de este algoritmo ya que no hay preconcepciones sobre el modelo ni sus hiperparámetros, por lo que es el mejor punto de partida para establecer una linea base.

                    En concreto se ha optado por realizar una búsqueda en cuadrícula (_grid search_), en la que definimos un espacio de búsqueda para los hiperparámetros y probamos las combinaciones para encontrar la mejor configuración.

                    Para implementar esta metodología se ha empleado la libreria _hyperopt_.

                    Se ha utilizado también validación cruzada para asegurar en la medida de lo posible que el modelo no presenta overfitting, con 5 muestras cruzadas.

                    ## Evaluación del modelo

                    A continuación pueden revisarse las diferentes gráficas que suelen emplearse para determinar el ajuste del modelo.
                    """)
    with gr.Row():
        desplegable = gr.Dropdown(
            choices=["Matriz de confusión", "Curva ROC", "Overfitting"],
            label="Selecciona el tipo de gráfico",
            value="Matriz de confusión"
        )
    with gr.Row():
        grafico = gr.Plot()

    desplegable.change(fn=generar_grafico_XGB, inputs=desplegable, outputs=grafico)

    with gr.Row():
        gr.Markdown(f"""
                    ### Matriz de confusión

                    La matriz de confusión nos permite evaluar el número de errores que comete el modelo en sus predicciones con el conjunto de prueba. Podemos dividir las predicciones en cuatro categorías:
                    - True Positives: En nuestro caso cuando el modelo acierta que el cliente quedó satisfecho. Cuadrante inferior derecho.
                    - True Negatives: En nuestro caso cuando el modelo acierta que el cliente quedó insatisfecho. Cuadrante superior izquierdo.
                    - False Positive: El modelo predice un cliente satisfecho cuando está insatisfecho. Cuadrante superior derecho.
                    - False Negatives: El modelo predice un cliente insatisfecho cuando está satisfecho. Cuadrante inferior izquierdo.

                    Como puede verse en la gráfica, el modelo presenta un número muy alto de TP y TN, lo cual indica un buen ajuste del modelo. 

                    Podemos operativizarlo de manera numérica calculando la precisión, calculada con estos valores numéricos.

                    En nuestro caso encontramos un ratio de precisión total del {np.round(accuracy, 2)}.

                    Podemos analizar también los valores del reporte de clasificación, donde se incluyen las medidas de exhaustividad (_recall_), proporción de verdaderos positivos entre los casos positivos (TP+NP), y el F1-score, media armónica de la precisión y la exhaustividad.

                    Reporte de clasificación: 

                    - Recall: Neutral o no satisfecho = 0.98 / Satisfecho = 0.94
                    - F1-score: Neutral o no satisfecho = 0.97 / Satisfecho = 0.96

                    Todos los valores obtenidos muestran valores por encima del 0.94, demostrando un gran ajuste del modelo.

                    ### Curva ROC

                    La curva ROC representa la compensación entre la tasa de True Positives y la tasa de False Positives. También puede conceptualizarse como una gráfica que muestra el poder estadístico como función del error tipo I.

                    Cuanto más se aproxime la curva a la esquina superior izquierda del gráfico, mejor consideraremos el modelo, ya que tiene una buena tasa de true positives.

                    Además de la gráfica, tenemos el valor AUC (_area under the curve_) como una cuantificación del rendimiento del modelo, basada en el área debajo de la curva. Este área representa la capacidad del modelo para distinguir entre positivos y negativos. A mayor valor, mayor capacidad de discriminación (del 0 al 1).

                    En el caso que nos atañe, ambos conceptos nos sirven para evaluar cómo de bien nuestro modelo es capaz de detectar la satisfacción de los clientes. 

                    En este caso atendiendo tanto a la gráfica, como al alto valor AUC (0.99). Podemos concluir que el modelo está discriminando de forma muy eficiente los casos de satisfacción.

                    ### Sobreajuste (Overfitting)

                    Por último, evaluamos el sobreajuste del modelo. El sobreajuste hace referencia al fenómeno por el cual un modelo se acostumbra demasiado a los datos de entrenamiento y no es capaz de generalizar el entrenamiento a nuevos datos de prueba.

                    Como puede verse en la gráfica de sobreajuste, para este modelo la tasa de acierto para el conjunto de entrenamiento y de prueba es muy similar, y, si calculamos el valor concreto del sobreajuste, encontramos que no llega a un 5% que es lo solicitado por el cliente.

                    Aún así, para aumentar la confianza en que el modelo no sobreajuste, se ha implementado validación cruzada, encontrándose el mismo resultado.

                    - Validación cruzada: {np.round(cv_scores, 2)}
                    - Media de las puntuaciones: {np.round(cv_scores.mean(), 2)}
                    """)

# Página 2

stack_model = joblib.load('../Modelos/ensemble_model.pkl')
cv = 5
train_accuracies = joblib.load('../Modelos/test_accuracies_kfold.pkl')
test_accuracies = joblib.load('../Modelos/train_accuracies_kfold.pkl')

y_pred = stack_model.predict(X_test)
y_prob = stack_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

class_report = classification_report(y_test, y_pred)


def generar_grafico_stack(tipo_grafico):
    plt.figure(figsize=(8,6))
    if tipo_grafico == "Matriz de confusión":
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Neutral o No satisfecho", "Satisfecho"], yticklabels=["Neutral o No Satisfecho", "Satisfecho"])
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de confusión')
        
    elif tipo_grafico == "Curva ROC":
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
    elif tipo_grafico == "Overfitting":
        plt.plot(range(1, cv+1), train_accuracies, label='Train Accuracy', marker='o')
        plt.plot(range(1, cv+1), test_accuracies, label='Test Accuracy', marker='o')

        plt.ylim(0.5, 1)
        plt.yticks(np.arange(0.50, 1.05, 0.05))

        plt.title('Train vs Test Accuracy en cada fold')
        plt.xlabel('Número de Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()  # Ajustar el diseño
    return plt.gcf()

def pagina2():
    with gr.Row():
        gr.Markdown("# Modelo de ML stacked")
    with gr.Row():
        gr.Markdown("""
                    ## Introducción

                    A continuación se presenta un modelo de prueba utilizando la técnica de ensemble de Stacking.

                    Esta técnica combina diferentes tipos de algoritmos de ML con el objetivo de conseguir resultados superiores a los que podría obtener un único algoritmo.

                    A pesar de que ya hemos utilizado una técnica de ensemble con el _gradient boosting_, hemos querido demostrar el funcionamiento de un modelo que lo combine con _stacking_.

                    Por ello, se ha decidido utilizar el modelo XGBoost que hemos podido ver que ofrece unos buenos resultados, en combinación con un modelo de regresión logística y un modelo de _random forest_ para comprobar si es posible mejorar el modelo. .

                    Para mejorar al máximo el rendimiento del modelo se han implementado los mismos hiperparámetros que en el modelo XGBoost:
                    - max_depth: Máxima profundidad del modelo ("ramas").
                    - gamma: Valor mínimo de pérdida para realizar una nueva partición del árbol. A mayor _gamma_ más conservador es el modelo.
                    - reg_alpha: Regularización L1, a mayor sea su valor más conservador es el modelo.
                    - min_child_weight: Suma mínima del peso en un "hijo". Si la partición del árbol resulta en una hoja nodo con una suma menor al valor especificado, el modelo dejará de realizar particiones.
                    - colsample_bytree: Submuestreo de columnas al construir el arbol. Especifica la fracción de columnas a submuestrear.
                    - eval_metric: La métrica para evaluar el modelo. Puede utilizarse varias, aquí usamos logloss, referida al valor negativo del log-likelihood.

                    Para determinar el mejor valor de los hiperparámetros se ha utilizado el mismo proceso que en el modelo con XGBoost, con algoritmo de búsqueda ingenuo. 

                    En concreto se ha optado por realizar una búsqueda en cuadrícula (_grid search_), en la que definimos un espacio de búsqueda para los hiperparámetros y probamos las combinaciones para encontrar la mejor configuración. Utilzando _hyperopt_.


                    Se ha utilizado también validación cruzada para asegurar en la medida de lo posible que el modelo no presenta overfitting, con 5 muestras cruzadas.

                    ## Evaluación del modelo

                    A continuación pueden revisarse las diferentes gráficas que suelen emplearse para determinar el ajuste del modelo.
                    """)
        
    with gr.Row():
        desplegable = gr.Dropdown(
            choices=["Matriz de confusión", "Curva ROC", "Overfitting"],
            label="Selecciona el tipo de gráfico",
            value="Matriz de confusión"
        )

    with gr.Row():
        grafico = gr.Plot()
    desplegable.change(fn=generar_grafico_stack, inputs=desplegable, outputs=grafico)

    with gr.Row():
        gr.Markdown(f"""
                ### Matriz de confusión

                La matriz de confusión nos permite evaluar el número de errores que comete el modelo en sus predicciones con el conjunto de prueba. Podemos dividir las predicciones en cuatro categorías:
                - True Positives: En nuestro caso cuando el modelo acierta que el cliente quedó satisfecho. Cuadrante inferior derecho.
                - True Negatives: En nuestro caso cuando el modelo acierta que el cliente quedó insatisfecho. Cuadrante superior izquierdo.
                - False Positive: El modelo predice un cliente satisfecho cuando está insatisfecho. Cuadrante superior derecho.
                - False Negatives: El modelo predice un cliente insatisfecho cuando está satisfecho. Cuadrante inferior izquierdo.

                Como puede verse en la gráfica, el modelo presenta un número muy alto de TP y TN, lo cual indica un buen ajuste del modelo. 

                Podemos operativizarlo de manera numérica calculando la precisión, calculada con estos valores numéricos.

                En nuestro caso encontramos un ratio de precisión total del {np.round(accuracy, 2)}.

                Podemos analizar también los valores del reporte de clasificación, donde se incluyen las medidas de exhaustividad (_recall_), proporción de verdaderos positivos entre los casos positivos (TP+NP), y el F1-score, media armónica de la precisión y la exhaustividad.

                Reporte de clasificación: 

                - Recall: Neutral o no satisfecho = 0.98 / Satisfecho = 0.94
                - F1-score: Neutral o no satisfecho = 0.97 / Satisfecho = 0.96

                Todos los valores obtenidos muestran valores por encima del 0.94, demostrando un gran ajuste del modelo.

                ### Curva ROC

                La curva ROC representa la compensación entre la tasa de True Positives y la tasa de False Positives. También puede conceptualizarse como una gráfica que muestra el poder estadístico como función del error tipo I.

                Cuanto más se aproxime la curva a la esquina superior izquierda del gráfico, mejor consideraremos el modelo, ya que tiene una buena tasa de true positives.

                Además de la gráfica, tenemos el valor AUC (_area under the curve_) como una cuantificación del rendimiento del modelo, basada en el área debajo de la curva. Este área representa la capacidad del modelo para distinguir entre positivos y negativos. A mayor valor, mayor capacidad de discriminación (del 0 al 1).

                En el caso que nos atañe, ambos conceptos nos sirven para evaluar cómo de bien nuestro modelo es capaz de detectar la satisfacción de los clientes. 

                En este caso atendiendo tanto a la gráfica, como al alto valor AUC (0.99). Podemos concluir que el modelo está discriminando de forma muy eficiente los casos de satisfacción.

                ### Sobreajuste (Overfitting)

                Por último, evaluamos el sobreajuste del modelo. El sobreajuste hace referencia al fenómeno por el cual un modelo se acostumbra demasiado a los datos de entrenamiento y no es capaz de generalizar el entrenamiento a nuevos datos de prueba.

                Como puede verse en la gráfica de sobreajuste, para este modelo la tasa de acierto para el conjunto de entrenamiento y de prueba es muy similar, y, si calculamos el valor concreto del sobreajuste, encontramos que no llega a un 5% que es lo solicitado por el cliente. Además se muestra que se repite para los 5 conjuntos de validación cruzada.
                """)

    
# Aplicación principal con navegación entre páginas
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Decision Tree with XGBoost"):
            pagina1()
        with gr.TabItem("Stacked Model"):
            pagina2()

# Iniciar la aplicación
demo.launch()