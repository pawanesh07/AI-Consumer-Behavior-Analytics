from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class SegmentationResult:
    rfm: pd.DataFrame
    model: KMeans
    scaler: StandardScaler


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Create Recency, Frequency, Monetary features for customer analytics."""
    if df.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary", "AvgOrderValue"])

    snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
        AvgOrderValue=("TotalAmount", "mean"),
        LastPurchase=("InvoiceDate", "max"),
    ).reset_index()
    return rfm


def assign_business_segment(row: pd.Series) -> str:
    """Human-readable segment based on customer behavior."""
    if row["Recency"] <= 45 and row["Frequency"] >= 8 and row["Monetary"] >= row.get("MonetaryMedian", 0):
        return "Champions"
    if row["Recency"] <= 60 and row["Frequency"] >= 5:
        return "Loyal Customers"
    if row["Recency"] <= 90 and row["Monetary"] >= row.get("MonetaryMedian", 0):
        return "Potential Loyalists"
    if row["Recency"] > 120 and row["Frequency"] <= 3:
        return "At Risk"
    return "Regular Customers"


def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> SegmentationResult:
    """Cluster customers using K-Means on RFM data."""
    rfm = build_rfm(df)
    if len(rfm) == 0:
        raise ValueError("No customer data available for segmentation.")

    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    n_clusters = min(n_clusters, len(rfm))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(scaled)

    monetary_median = rfm["Monetary"].median()
    rfm["MonetaryMedian"] = monetary_median
    rfm["Segment"] = rfm.apply(assign_business_segment, axis=1)
    rfm.drop(columns=["MonetaryMedian"], inplace=True)
    return SegmentationResult(rfm=rfm, model=model, scaler=scaler)
