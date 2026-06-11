from __future__ import annotations

import pandas as pd


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Month", as_index=False)["TotalAmount"].sum().rename(columns={"TotalAmount": "Revenue"})


def revenue_by_country(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Country", as_index=False)["TotalAmount"].sum().sort_values("TotalAmount", ascending=False)


def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Category", as_index=False).agg(
        Revenue=("TotalAmount", "sum"),
        QuantitySold=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"),
    ).sort_values("Revenue", ascending=False)


def kpi_summary(df: pd.DataFrame) -> dict:
    return {
        "total_revenue": float(df["TotalAmount"].sum()),
        "total_orders": int(df["InvoiceNo"].nunique()),
        "total_customers": int(df["CustomerID"].nunique()),
        "avg_order_value": float(df.groupby("InvoiceNo")["TotalAmount"].sum().mean()),
        "total_products": int(df["StockCode"].nunique()),
    }
