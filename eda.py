import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar el archivo CSV
file_path = '/Users/sunbay/Desktop/ML-grupo5/ML-Grupo5/airline_passenger_satisfaction.csv'
df = pd.read_csv(file_path)

# Mostrar las primeras filas del DataFrame
print(df.head())

# Resumen de los datos
print(df.info())

# Estadísticas descriptivas básicas
print(df.describe())

# Verificar valores nulos
print(df.isnull().sum())

# Eliminar filas con valores nulos para simplificar el análisis de correlación
df_clean = df.dropna()

# Seleccionar solo columnas numéricas para la matriz de correlación
numeric_columns = df_clean.select_dtypes(include=['float64', 'int64'])

# Histograma de la variable 'Age'
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribución de la Edad de los Pasajeros')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Histograma de la variable 'Flight Distance'
plt.figure(figsize=(10, 6))
plt.hist(df['Flight Distance'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Distribución de la Distancia de Vuelo')
plt.xlabel('Distancia de Vuelo')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Histograma de la variable 'Inflight wifi service'
plt.figure(figsize=(10, 6))
plt.hist(df['Inflight wifi service'], bins=5, color='lightcoral', edgecolor='black')
plt.title('Distribución del Servicio de Wifi a Bordo')
plt.xlabel('Calificación del Servicio de Wifi')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Histograma de la variable 'Satisfaction' para observar la satisfacción general
plt.figure(figsize=(10, 6))
df['satisfaction'].value_counts().plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('Distribución de la Satisfacción de los Pasajeros')
plt.xlabel('Satisfacción')
plt.ylabel('Número de Pasajeros')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# Filtrar por género
male_ages = df[df['Gender'] == 'Male']['Age']
female_ages = df[df['Gender'] == 'Female']['Age']

# Crear subplots para histograma separados
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Histograma para género masculino
axs[0].hist(male_ages, bins=30, color='blue', edgecolor='black')
axs[0].set_title('Distribución de la Edad (Masculino)')
axs[0].set_xlabel('Edad')
axs[0].set_ylabel('Frecuencia')
axs[0].grid(True)

# Histograma para género femenino
axs[1].hist(female_ages, bins=30, color='pink', edgecolor='black')
axs[1].set_title('Distribución de la Edad (Femenino)')
axs[1].set_xlabel('Edad')
axs[1].grid(True)

plt.tight_layout()
plt.show()


# Boxplots de la Edad según Satisfacción y Clase de Vuelo
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Customer Type', hue='satisfaction', palette='viridis')
plt.title('Satisfacción según Tipo de Cliente')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Cantidad')
plt.legend(title='Satisfacción')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Class', hue='satisfaction', palette='viridis')
plt.title('Satisfacción según Clase de Vuelo')
plt.xlabel('Clase de Vuelo')
plt.ylabel('Cantidad')
plt.legend(title='Satisfacción')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Class', y='Age', hue='satisfaction', palette='viridis')
plt.title('Distribución de la Edad según Clase de Vuelo y Satisfacción')
plt.xlabel('Clase de Vuelo')
plt.ylabel('Edad')
plt.legend(title='Satisfacción')
plt.show()

# Relación entre la Distancia del Vuelo y la Satisfacción
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='satisfaction', y='Flight Distance', palette='Set3')
plt.title('Relación entre la Distancia del Vuelo y la Satisfacción')
plt.xlabel('Satisfacción')
plt.ylabel('Distancia del Vuelo')
plt.show()
