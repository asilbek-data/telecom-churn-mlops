"""FastAPI service that serves the trained churn model."""

import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import Customer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(ROOT, "models", "model.pkl"))

app = FastAPI(title="Churn Scoring API", version="1.0")
model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"service": "Churn Scoring API", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}


@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "will_churn": bool(proba >= 0.5),
        "risk": "high" if proba >= 0.5 else "low",
    }
