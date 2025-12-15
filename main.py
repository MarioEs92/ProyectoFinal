# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Literal

# --- 1. Definición del Modelo de Entrada (Schema) ---
# Usamos las features del German Credit Data
class CreditApplicant(BaseModel):
    # Features Categóricas (usaremos Tipos Literales para validación estricta)
    Purpose: Literal['car', 'furniture/equipment', 'radio/tv', 'domestic appliances', 'repairs', 'education', 'vacation', 'retraining', 'business', 'other']
    Sex: Literal['male', 'female']
    Housing: Literal['rent', 'own', 'free']
    Saving_accounts: Literal['little', 'moderate', 'quite rich', 'rich', np.nan] # Incluimos NaN
    Checking_account: Literal['little', 'moderate', 'rich', np.nan] # Incluimos NaN
    # Features Numéricas
    Age: int
    Job: int
    Credit_amount: int
    Duration: int # En meses
# --- 2. Inicialización y Carga del Pipeline ---
app = FastAPI(title="Credit Risk Scoring API - German Credit Data")

PIPELINE_PATH = 'risk_model_pipeline.pkl'
FEATURE_NAMES_PATH = 'original_feature_names.pkl'
try:
    with open(PIPELINE_PATH, 'rb') as file:
        model_pipeline = pickle.load(file)
    print(f"Pipeline cargado exitosamente desde: {PIPELINE_PATH}")

    with open(FEATURE_NAMES_PATH, 'rb') as file:
        original_feature_names = pickle.load(file)
    print("Nombres de features originales cargados.")

except Exception as e:
    print(f"ERROR al cargar el pipeline o las features: {e}")
    # En un entorno real, la API debería fallar si el modelo no carga
    raise RuntimeError(f"Fallo al cargar el modelo necesario: {e}")

# --- 3. Endpoint de Predicción ---
@app.post("/predict/risk")
async def predict_risk(applicant: CreditApplicant):
    # 3.1. Convertir el objeto Pydantic a un diccionario y luego a un DataFrame
    applicant_dict = applicant.model_dump()
    input_df = pd.DataFrame([applicant_dict])

    # ----------------------------------------------------------------------
    # FIX: RENOMBRAR COLUMNAS Y LIMPIAR LA LISTA DE FEATURES DEL MODELO
    # ----------------------------------------------------------------------

    # 1. Renombrar las columnas de input_df (de _ a espacio) para que coincidan con el .pkl
    column_mapping = {
        'Saving_accounts': 'Saving accounts',
        'Checking_account': 'Checking account',
        'Credit_amount': 'Credit amount',
        # Si tienes más columnas con espacios en el modelo original, añádelas aquí:
        # Ejemplo: 'Job_Type': 'Job Type'
    }
    # Aplicar el renombramiento
    input_df.rename(columns=column_mapping, inplace=True)
    
    # 2. Limpiar la lista de nombres de features. Eliminar el índice basura ('Unnamed: 0')
    # Y cualquier otra feature que el JSON NO provee, pero que está en original_feature_names
    
    # CRÍTICO: Asegurarse de que 'Unnamed: 0' se elimina de la lista de features esperadas
    clean_feature_names = [f for f in original_feature_names if f != 'Unnamed: 0']

    # 3.2. Asegurar el orden de las columnas con la lista limpia
    try:
        # Ahora el DataFrame debería contener todos los nombres de 'clean_feature_names'
        input_df = input_df[clean_feature_names]
    except KeyError as e:
        # Si esto aún falla, significa que el JSON no provee una columna esperada
        raise HTTPException(status_code=400, detail=f"Error interno de features. Faltan datos requeridos: {e}")
    
    # 3.3. Predecir
    proba = model_pipeline.predict_proba(input_df)[0][1]

    # 3.4. Regla de Negocio (Umbral)
    RISK_THRESHOLD = 0.50  # Umbral base (Ajustable)
    decision = "RECHAZADO" if proba >= RISK_THRESHOLD else "APROBADO"

    return {
        "status": "success",
        "risk_score_percent": round(proba * 100, 2),
        "prediction": decision,
        "details": f"Probabilidad de incumplimiento (default) calculada: {round(proba, 4)}"
    }
 
# --- 4. Endpoint de Bienvenida ---
@app.get("/")
async def root():
    return {"message": "Credit Risk Scoring API (German Data) is running.",
            "model_status": "Ready",
            "model_type": "Logistic Regression Pipeline"}