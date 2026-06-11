from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "online_retail_sample.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "cleaned_customer_data.csv"


def format_currency(value: float) -> str:
    return f"₹{value:,.0f}"


def add_project_root_to_path(file_path: str, levels_up: int = 1) -> Path:
    root = Path(file_path).resolve().parents[levels_up]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def generate_insights(df: pd.DataFrame, rfm: pd.DataFrame | None = None, churn_df: pd.DataFrame | None = None) -> list[str]:
    insights: list[str] = []
    if df.empty:
        return ["No data available. Upload or use the sample dataset."]

    top_month = df.groupby("Month")["TotalAmount"].sum().idxmax()
    top_country = df.groupby("Country")["TotalAmount"].sum().idxmax()
    top_product = df.groupby("Description")["TotalAmount"].sum().idxmax()
    insights.append(f"Highest revenue month is {top_month}. Focus marketing campaigns around similar seasonal demand.")
    insights.append(f"{top_country} contributes the highest revenue. This region should be prioritized for retention offers.")
    insights.append(f"Top revenue product is {top_product}. Bundle it with related items to increase average order value.")

    if rfm is not None and not rfm.empty:
        at_risk = int((rfm["Segment"] == "At Risk").sum())
        champions = int((rfm["Segment"] == "Champions").sum())
        insights.append(f"There are {champions} champion customers and {at_risk} at-risk customers based on RFM behavior.")

    if churn_df is not None and not churn_df.empty and "RiskLevel" in churn_df:
        high_risk = int((churn_df["RiskLevel"] == "High").sum())
        insights.append(f"{high_risk} customers are currently in high churn-risk category. Send personalized offers to retain them.")
    return insights
