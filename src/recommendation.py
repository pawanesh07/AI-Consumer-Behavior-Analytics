from __future__ import annotations

import pandas as pd


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.groupby(["StockCode", "Description"])
        .agg(QuantitySold=("Quantity", "sum"), Revenue=("TotalAmount", "sum"), Orders=("InvoiceNo", "nunique"))
        .reset_index()
        .sort_values(["Revenue", "QuantitySold"], ascending=False)
        .head(n)
    )


def recommend_for_customer(df: pd.DataFrame, customer_id: str, n: int = 5) -> pd.DataFrame:
    """Simple item recommendation using products bought by similar customers."""
    customer_id = str(customer_id)
    all_products = top_products(df, n=50)
    if customer_id not in set(df["CustomerID"].astype(str)):
        return all_products.head(n).assign(Reason="Popular product")

    customer_products = set(df.loc[df["CustomerID"].astype(str) == customer_id, "StockCode"])
    similar_customers = df[df["StockCode"].isin(customer_products)]["CustomerID"].astype(str).unique()
    candidates = df[
        (df["CustomerID"].astype(str).isin(similar_customers)) &
        (~df["StockCode"].isin(customer_products))
    ]
    recs = top_products(candidates, n=n)
    if recs.empty:
        recs = all_products.head(n)
        recs["Reason"] = "Popular product"
    else:
        recs["Reason"] = "Bought by similar customers"
    return recs
