"""CI tests: the data loads, the pipeline trains, and it predicts sanely.

These run in GitHub Actions on every push. A failure blocks the change.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from preprocess import build_preprocessor, load_data  # noqa: E402

DATA = os.getenv("DATA_PATH", "data/telco_churn.csv")


def test_data_loads():
    X, y = load_data(DATA)
    assert len(X) == len(y) > 1000
    assert set(y.unique()) <= {0, 1}
    assert "customerID" not in X.columns
    assert X["TotalCharges"].dtype.kind == "f"


def test_pipeline_trains_and_beats_baseline():
    X, y = load_data(DATA)
    pre = build_preprocessor(X)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    assert auc > 0.78, f"model regressed: ROC-AUC={auc:.3f}"


def test_high_risk_scores_above_low_risk():
    """A new month-to-month fiber customer must score higher than a
    long-tenure two-year customer. Catches a silently broken pipeline."""
    import joblib
    import pandas as pd

    model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    if not os.path.exists(model_path):
        import pytest

        pytest.skip("models/model.pkl not built yet")

    model = joblib.load(model_path)
    base = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "PaperlessBilling": "Yes",
    }
    high = {
        **base,
        "tenure": 2,
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
    }
    low = {
        **base,
        "tenure": 70,
        "InternetService": "DSL",
        "Contract": "Two year",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 60.0,
        "TotalCharges": 4200.0,
    }
    p_high = model.predict_proba(pd.DataFrame([high]))[0, 1]
    p_low = model.predict_proba(pd.DataFrame([low]))[0, 1]
    assert p_high > 0.5 > p_low, f"high={p_high:.3f} low={p_low:.3f}"
