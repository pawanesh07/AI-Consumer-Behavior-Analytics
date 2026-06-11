from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.segmentation import build_rfm


@dataclass
class ChurnResult:
    customer_churn: pd.DataFrame
    model: RandomForestClassifier
    accuracy: float
    report: str


def create_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create churn features and a practical churn label for demo/portfolio use."""
    rfm = build_rfm(df)
    if rfm.empty:
        return rfm

    # Label customers as churn-risk when they have not purchased recently and have low engagement.
    recency_cutoff = rfm["Recency"].quantile(0.70)
    frequency_cutoff = rfm["Frequency"].quantile(0.40)
    monetary_cutoff = rfm["Monetary"].quantile(0.35)
    rfm["Churn"] = (
        (rfm["Recency"] >= recency_cutoff) &
        (rfm["Frequency"] <= frequency_cutoff) &
        (rfm["Monetary"] <= monetary_cutoff)
    ).astype(int)
    return rfm


def train_churn_model(df: pd.DataFrame) -> ChurnResult:
    data = create_churn_features(df)
    if len(data) < 10 or data["Churn"].nunique() < 2:
        data["ChurnProbability"] = 0.0
        data["RiskLevel"] = "Low"
        return ChurnResult(data, RandomForestClassifier(random_state=42), 0.0, "Not enough variation to train model.")

    X = data[["Recency", "Frequency", "Monetary", "AvgOrderValue"]]
    y = data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    probabilities = model.predict_proba(X)[:, 1]
    data["ChurnProbability"] = (probabilities * 100).round(2)
    data["RiskLevel"] = pd.cut(
        data["ChurnProbability"],
        bins=[-1, 35, 70, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return ChurnResult(data.sort_values("ChurnProbability", ascending=False), model, float(accuracy), report)
