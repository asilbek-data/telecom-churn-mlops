"""Streamlit UI: fill in a customer, get a churn score.

Primary path : POST the customer to the FastAPI /predict endpoint (API_URL).
Fallback path: if the API is unreachable (Render free tier sleeps), score the
customer locally with the same models/model.pkl. The deployed demo therefore
never shows a dead page.
"""

import os

import joblib
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_local_model():
    """Loaded once and reused across reruns."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def score_customer(payload):
    """Return (result_dict, source). Try the API first, fall back to local."""
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), "API"
    except Exception:
        model = load_local_model()
        if model is None:
            return None, "none"
        proba = float(model.predict_proba(pd.DataFrame([payload]))[0, 1])
        return {
            "churn_probability": round(proba, 4),
            "will_churn": bool(proba >= 0.5),
            "risk": "high" if proba >= 0.5 else "low",
        }, "local model"


st.title("📉 Telecom Churn Predictor")
st.caption("Enter a customer's details to estimate the chance they will leave.")

YN = ["Yes", "No"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", [0, 1])
    partner = st.selectbox("Partner", YN)
    dependents = st.selectbox("Dependents", YN, index=1)
    tenure = st.slider("Tenure (months)", 0, 72, 5)

    st.subheader("Contract & billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", YN)
    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 89.9, step=1.0)
    total = st.number_input("Total charges", 0.0, 10000.0, 450.5, step=10.0)

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone service", YN)
    lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"], index=1)
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], index=1)

    no_net = internet == "No"
    net_opts = ["No internet service"] if no_net else YN
    idx = 0 if no_net else 1

    security = st.selectbox("Online security", net_opts, index=idx)
    backup = st.selectbox("Online backup", net_opts, index=idx)
    protection = st.selectbox("Device protection", net_opts, index=idx)
    support = st.selectbox("Tech support", net_opts, index=idx)
    tv = st.selectbox("Streaming TV", net_opts, index=idx)
    movies = st.selectbox("Streaming movies", net_opts, index=idx)

payload = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": lines if phone == "Yes" else "No phone service",
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    out, source = score_customer(payload)

    if out is None:
        st.error(f"No API at {API_URL} and no local model at {MODEL_PATH}.")
    else:
        p = out["churn_probability"]
        st.metric("Churn probability", f"{p * 100:.1f}%")
        st.progress(min(p, 1.0))
        if out["will_churn"]:
            st.error(f"Risk: {out['risk'].upper()} — likely to leave.")
        else:
            st.success(f"Risk: {out['risk'].upper()} — likely to stay.")
        st.caption(f"Scored via: {source}")

with st.expander("Request payload sent to /predict"):
    st.json(payload)

st.caption(
    "Model: scikit-learn pipeline (ROC-AUC 0.84 on the IBM Telco dataset, "
    "7,043 customers) · tracked with MLflow · served by FastAPI."
)
