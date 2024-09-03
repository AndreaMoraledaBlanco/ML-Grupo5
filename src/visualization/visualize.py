import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt

def plot_shap_summary(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig('reports/figures/shap_summary.png')
    plt.close()

def plot_feature_importance(model, X, n_top_features=10):
    """
    Genera un gráfico de barras de la importancia de las características.
    """
    print("Generando gráfico de importancia de características...")
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(n_top_features))
    plt.title(f'Top {n_top_features} Características Más Importantes')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()

def plot_roc_curve(model, X_test, y_test):
    """
    Genera la curva ROC.
    """
    print("Generando curva ROC...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png')
    plt.close()

def plot_confusion_matrix(conf_matrix):
    """
    Genera una visualización de la matriz de confusión.
    """
    print("Generando matriz de confusión...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()

def plot_permutation_importance(model, X_test, y_test, n_top_features=10):
    """
    Genera un gráfico de la importancia de las características basado en permutación.
    """
    print("Generando gráfico de importancia por permutación...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importances = pd.DataFrame({
        'feature': X_test.columns,
        'importance': perm_importance.importances_mean
    })
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(n_top_features))
    plt.title(f'Top {n_top_features} Características (Importancia por Permutación)')
    plt.tight_layout()
    plt.savefig('reports/figures/permutation_importance.png')
    plt.close()

def visualize_results(model, X_test, y_test, conf_matrix):
    """
    Genera todas las visualizaciones.
    """
    plot_feature_importance(model, X_test)
    plot_roc_curve(model, X_test, y_test)
    plot_confusion_matrix(conf_matrix)
    plot_permutation_importance(model, X_test, y_test)

    print("Visualizaciones guardadas en la carpeta 'reports/figures/'")

if __name__ == "__main__":
    # Este código se ejecutará si ejecutas este script directamente
    print("Este script está diseñado para ser importado y usado en train_model.py")
    print("Para generar visualizaciones, ejecuta train_model.py")