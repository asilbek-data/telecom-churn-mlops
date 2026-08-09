"""Data loading + the feature pipeline. Shared by training AND serving.

Keeping this in one place is what prevents train/serve skew: the API uses the
exact same transformations that were used at training time, because the fitted
ColumnTransformer is stored inside models/model.pkl.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "Churn"
DROP = ["customerID"]


def load_data(csv_path):
    """Read the CSV, fix TotalCharges, split into features X and target y (0/1)."""
    df = pd.read_csv(csv_path)
    # brand-new customers have a blank TotalCharges -> make numeric, fill with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor(X):
    """Scale numbers, one-hot encode categories.

    handle_unknown='ignore' means a category the model never saw at training
    time will never crash the API at prediction time.
    """
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes("object").columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )
