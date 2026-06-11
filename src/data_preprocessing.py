from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load CSV or Excel transaction data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make common e-commerce dataset column names compatible with this project."""
    df = df.copy()
    rename_map = {
        "Invoice": "InvoiceNo",
        "invoice_no": "InvoiceNo",
        "invoice": "InvoiceNo",
        "product_id": "StockCode",
        "ProductID": "StockCode",
        "Product": "Description",
        "product_name": "Description",
        "Customer ID": "CustomerID",
        "customer_id": "CustomerID",
        "Date": "InvoiceDate",
        "date": "InvoiceDate",
        "Price": "UnitPrice",
        "price": "UnitPrice",
        "qty": "Quantity",
        "quantity": "Quantity",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)
    return df


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transaction data and add derived analytics columns."""
    df = normalize_columns(df)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing) +
            ". Use the included sample dataset or rename your columns."
        )

    df = df.copy()
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["CustomerID", "Description", "InvoiceNo", "InvoiceDate"])

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])

    # Remove returns/cancellations and invalid prices.
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    df["CustomerID"] = df["CustomerID"].astype(float).astype(int).astype(str)
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["OrderDate"] = df["InvoiceDate"].dt.date

    if "Category" not in df.columns:
        df["Category"] = "General"

    return df.reset_index(drop=True)


def save_clean_data(raw_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Load, clean, save and return the processed dataframe."""
    df = load_data(raw_path)
    clean_df = clean_retail_data(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    return clean_df
