import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import shap
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    logging.info(f"Cargando datos desde {filepath}")
    data = pd.read_csv(filepath)
    data['satisfaction'] = data['satisfaction'].astype(int)
    logging.info(f"Forma del DataFrame: {data.shape}")
    logging.info(f"Columnas: {data.columns.tolist()}")
    logging.info(f"Distribución de 'satisfaction':\n{data['satisfaction'].value_counts(normalize=True)}")
    return data

def create_interaction_features(X):
    logging.info("Creando características de interacción...")
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    if 'Inflight wifi service' in X.columns and 'Inflight entertainment' in X.columns:
        X['wifi_entertainment'] = X['Inflight wifi service'] * X['Inflight entertainment']
    
    service_columns = ['Food and drink', 'Inflight service', 'On-board service']
    if all(col in X.columns for col in service_columns):
        X['total_service'] = X[service_columns].sum(axis=1)
    
    return X

def create_models():
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    return models

def create_pipeline(model):
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('feature_engineer', FunctionTransformer(create_interaction_features, validate=False)),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

def optimize_hyperparameters(pipeline, X, y):
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, 
                                       n_iter=20, cv=cv, scoring='roc_auc', 
                                       n_jobs=-1, random_state=42, verbose=1)
    
    random_search.fit(X, y)
    logging.info(f"Mejores hiperparámetros: {random_search.best_params_}")
    return random_search.best_estimator_

def plot_learning_curve(estimator, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title("Learning Curve")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def plot_feature_importance(model, X):
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = X.columns

        plt.figure(figsize=(10, 6))
        plt.title("Importancia de características")
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png')
        plt.close()

def plot_shap_summary(model, X):
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig('reports/figures/shap_summary.png')
    plt.close()

def save_model(model, filepath):
    joblib.dump(model, filepath)
    logging.info(f"Modelo guardado en {filepath}")

def calculate_overfitting(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfitting = (train_score - test_score) / train_score * 100
    logging.info(f"Overfitting: {overfitting:.2f}%")
    return overfitting

def main():
    logging.info("Iniciando el entrenamiento del modelo...")
    
    try:
        # Cargar datos
        data = load_data("data/processed/clean_airline_satisfaction.csv")
        X = data.drop("satisfaction", axis=1)
        y = data["satisfaction"]
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Crear y evaluar modelos
        models = create_models()
        results = {}
        
        for name, model in models.items():
            logging.info(f"Evaluando {name}...")
            pipeline = create_pipeline(model)
            try:
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
                results[name] = {'cv_score': np.mean(cv_scores)}
                logging.info(f"{name} - CV ROC AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            except Exception as e:
                logging.error(f"Error al evaluar {name}: {str(e)}")
                logging.error(traceback.format_exc())
        
        if not results:
            logging.error("No se pudo evaluar ningún modelo. Verificar los datos y la configuración.")
            return
        
        # Seleccionar el mejor modelo
        best_model_name = max(results, key=lambda x: results[x]['cv_score'])
        logging.info(f"Mejor modelo: {best_model_name}")
        
        # Optimizar el mejor modelo
        best_pipeline = create_pipeline(models[best_model_name])
        optimized_model = optimize_hyperparameters(best_pipeline, X_train, y_train)
        
        # Evaluar el modelo optimizado
        final_scores = evaluate_model(optimized_model, X_test, y_test)
        for metric, score in final_scores.items():
            logging.info(f"{metric}: {score:.4f}")
        
        # Calcular overfitting
        overfitting = calculate_overfitting(optimized_model, X_train, y_train, X_test, y_test)
        if overfitting < 5:
            logging.info("El overfitting está dentro del rango aceptable (<5%)")
        else:
            logging.warning("El overfitting es superior al 5%, considera aplicar más regularización")
        
        # Generar visualizaciones
        plot_learning_curve(optimized_model, X, y)
        plt.savefig('reports/figures/learning_curve.png')
        plt.close()
        
        plot_feature_importance(optimized_model, X)
        plot_shap_summary(optimized_model, X_test)
        
        # Guardar el modelo final
        save_model(optimized_model, "models/best_model.joblib")
        
        logging.info("Entrenamiento del modelo completado.")
    
    except Exception as e:
        logging.error(f"Se produjo un error: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()