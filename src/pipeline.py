"""Phase 2 (cleaning) + Phase 3 (feature engineering) for Telco churn.

Returns a fully numeric, one-hot encoded feature matrix usable by both linear and
tree models. Imported by src/model.py.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/telco_churn.csv")
TARGET = "Churn"
ID = "customerID"
SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "MultipleLines"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # classic gotcha: TotalCharges has 11 blank strings (tenure==0 new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    # normalise the "No internet/phone service" third level -> "No" is kept as its own signal,
    # but we also add explicit flags in features. Ensure plain python str dtype (arrow -> object).
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # number of add-on services actually subscribed
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["auto_payment"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    df["streaming_count"] = ((df["StreamingTV"] == "Yes").astype(int)
                             + (df["StreamingMovies"] == "Yes").astype(int))
    # billing dynamics
    df["charges_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["monthly_x_tenure"] = df["MonthlyCharges"] * df["tenure"]
    df["avg_monthly_billed"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
    df["monthly_vs_avg"] = df["MonthlyCharges"] - df["avg_monthly_billed"]
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    return df


def build(path=DATA):
    df = add_features(clean(pd.read_csv(path)))
    y = (df[TARGET] == "Yes").astype(int).values
    df = df.drop(columns=[ID, TARGET])
    # one-hot encode remaining categoricals; keep numeric as-is
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    X = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    X = X.astype(float)
    return X, y, list(X.columns)


if __name__ == "__main__":
    X, y, feats = build()
    print(f"X {X.shape} | churn rate {y.mean():.3f} | features {len(feats)}")
    print("engineered:", [c for c in feats if c in (
        "num_services", "has_internet", "is_month_to_month", "auto_payment", "streaming_count",
        "charges_per_tenure", "monthly_x_tenure", "avg_monthly_billed", "monthly_vs_avg", "is_new_customer")])
