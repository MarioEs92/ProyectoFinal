# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import numpy as np

# --- 1. CONFIGURACIÓN ---
MODEL_FILE_NAME = 'credit_risk_model.pkl'
DATA_FILE_NAME = 'german_credit_data.csv'

# --- 2. CARGAR Y PREPROCESAR DATOS ---
def load_and_preprocess_data(file_path):
    """Carga los datos y realiza el preprocesamiento necesario."""
    print(f"Cargando datos desde: {file_path}")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {file_path}. Asegúrate de descargarlo.")
        return None, None

    # El dataset tiene 21 columnas (features) y 1 columna target ('Risk')
    
    # Mapeo de la variable objetivo (Target)
    # 'Risk': 1 = Mal Crédito (Default), 0 = Buen Crédito
    # El dataset de Kaggle usa 0 para Good y 1 para Bad, lo invertiremos para que 1 sea el evento de riesgo (Default)
    data['Risk'] = data['Risk'].apply(lambda x: 1 if x == 1 else 0)
    
    # 2.1 Codificación de Variables Categóricas
    # Seleccionamos las columnas categóricas (object)
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    print("Codificando variables categóricas...")
    for col in categorical_cols:
        le = LabelEncoder()
        # Ajustamos y transformamos para cada columna
        data[col] = le.fit_transform(data[col])

    # 2.2 Normalización de Variables Numéricas (opcional, pero recomendado)
    numerical_cols = data.select_dtypes(include=[np.number]).columns.drop('Risk')
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # 2.3 Definición de Features (X) y Target (y)
    X = data.drop('Risk', axis=1)
    y = data['Risk']

    print(f"Datos preprocesados. Dimensiones: {X.shape}")
    return X, y

# --- 3. ENTRENAMIENTO DEL MODELO ---
def train_and_save_model(X, y):
    """Entrena el modelo de Bosque Aleatorio y lo guarda."""
    
    # Dividir datos en entrenamiento (70%) y prueba (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Inicializar el modelo: Bosque Aleatorio
    # El parámetro class_weight='balanced' es útil para datasets desbalanceados
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced'
    )

    print("Iniciando entrenamiento del modelo...")
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    # 4. Evaluación del Modelo
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Métricas clave (Recordar: queremos alto Recall para identificar los casos de riesgo)
    print("\n--- Métricas de Evaluación (Conjunto de Prueba) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall (Sensibilidad): {recall_score(y_test, y_pred):.4f}") # Importante para Credit Risk
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # 5. Guardar el Modelo
    with open(MODEL_FILE_NAME, 'wb') as file:
        pickle.dump(model, file)
    print(f"\nModelo guardado exitosamente como: {MODEL_FILE_NAME}")

if __name__ == "__main__":
    X, y = load_and_preprocess_data(DATA_FILE_NAME)
    
    if X is not None:
        # Una vez que tengas el modelo entrenado, debes asegurarte de que
        # las features de entrada de la API (model_structure.py) sean las mismas.
        # En este caso, la API debe recibir las 20 columnas del dataset German Credit
        print("\nLista de Features usadas para el modelo (deben coincidir con la API):")
        print(list(X.columns))
        
        train_and_save_model(X, y)