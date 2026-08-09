"""Train churn models, track every run in MLflow, save the best pipeline.

Run:  python src/train.py
Then: mlflow ui   (open http://127.0.0.1:5000 to compare the runs)
"""

import os
import sys

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import build_preprocessor, load_data  # noqa: E402

DATA_PATH = os.getenv("DATA_PATH", "data/telco_churn.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")


def get_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


def main():
    X, y = load_data(DATA_PATH)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_experiment("churn")
    best_pipe, best_auc, best_name = None, -1.0, None

    for name, clf in get_models().items():
        with mlflow.start_run(run_name=name):
            # a fresh preprocessor per run so nothing leaks between models
            pipe = Pipeline([("pre", build_preprocessor(X)), ("clf", clf)])
            pipe.fit(Xtr, ytr)

            proba = pipe.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, proba)
            ap = average_precision_score(yte, proba)
            f1 = f1_score(yte, pipe.predict(Xte))

            mlflow.log_param("model", name)
            mlflow.log_metrics({"roc_auc": auc, "pr_auc": ap, "f1": f1})
            mlflow.sklearn.log_model(pipe, name="model")
            print(f"{name:14} ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}  F1={f1:.4f}")

            if auc > best_auc:
                best_pipe, best_auc, best_name = pipe, auc, name

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"\nBEST: {best_name}  (ROC-AUC={best_auc:.4f})  ->  saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
