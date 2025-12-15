# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pickle
import os

# --- 1. Cargar y Preprocesar Datos (Dataset de German Credit Data) ---
try:
    # Cargar el dataset sin index_col=0 para asegurar que todas las columnas estén presentes
    df = pd.read_csv('german_credit_data.csv')
except FileNotFoundError:
    print("ERROR: El archivo 'german_credit_data.csv' no fue encontrado.")
    print("Por favor, descárgalo de Kaggle o usa el comando '!wget <URL>' en entornos como Colab/Jupyter.")
    exit()

# La variable objetivo 'Risk' debe ser numérica (0: Buen Crédito, 1: Mal Crédito/Incumplimiento)
# El dataset ya está en 0 y 1.

# Definir las variables Categóricas y Numéricas
# El preprocesamiento es crucial para la serialización.
categorical_features = ['Purpose', 'Sex', 'Housing', 'Saving accounts', 'Checking account']
numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']

# Definir el preprocesador: aplica OneHotEncoder a las categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Mantiene las columnas numéricas
)
# --- 2. Preparación para Entrenamiento ---
X = df.drop('Risk', axis=1)
y = df['Risk']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# --- 3. Creación del Pipeline (Preprocesamiento + Modelo) ---
# El Pipeline garantiza que el preprocesamiento se aplique siempre de la misma forma.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

print("Iniciando entrenamiento del Pipeline (Preprocesamiento + Regresión Logística)...")
model_pipeline.fit(X_train, y_train)
# --- 4. Evaluación ---
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC en el conjunto de prueba: {auc_score:.4f}")
print("¡Pipeline entrenado exitosamente!")
# --- 5. Serialización del Pipeline y Nombres de Features ---

# A. Serializar el Pipeline COMPLETO (preprocesador + modelo)
pipeline_filename = 'risk_model_pipeline.pkl'
with open(pipeline_filename, 'wb') as file:
    pickle.dump(model_pipeline, file)

print(f"Pipeline serializado y guardado como {pipeline_filename}")

# B. Guardar la lista de features originales para la validación de la API
original_feature_names = list(X.columns)
with open('original_feature_names.pkl', 'wb') as file:
    pickle.dump(original_feature_names, file)
print("Nombres de features originales guardados como original_feature_names.pkl")
