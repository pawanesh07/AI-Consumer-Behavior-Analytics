from __future__ import annotations

from pathlib import Path

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

    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make common e-commerce dataset column names compatible with this project."""
    df = df.copy()

    # Remove extra spaces from column names
    df.columns = df.columns.astype(str).str.strip()

    rename_map = {
        # Invoice / Order ID
        "Invoice": "InvoiceNo",
        "Invoice No": "InvoiceNo",
        "Invoice_No": "InvoiceNo",
        "invoice_no": "InvoiceNo",
        "invoice": "InvoiceNo",
        "Order ID": "InvoiceNo",
        "OrderID": "InvoiceNo",
        "Order Id": "InvoiceNo",
        "Transaction ID": "InvoiceNo",
        "Transaction_ID": "InvoiceNo",
        "TransactionID": "InvoiceNo",

        # Product / Stock Code
        "Stock Code": "StockCode",
        "stock_code": "StockCode",
        "Product Code": "StockCode",
        "ProductCode": "StockCode",
        "Product ID": "StockCode",
        "ProductID": "StockCode",
        "product_id": "StockCode",
        "Item ID": "StockCode",
        "ItemID": "StockCode",

        # Product Description
        "Product": "Description",
        "Product Name": "Description",
        "product_name": "Description",
        "Item": "Description",
        "Item Purchased": "Description",
        "Item Name": "Description",
        "Product Description": "Description",

        # Quantity
        "Qty": "Quantity",
        "qty": "Quantity",
        "quantity": "Quantity",
        "Quantity Purchased": "Quantity",

        # Date
        "Date": "InvoiceDate",
        "date": "InvoiceDate",
        "Invoice Date": "InvoiceDate",
        "Invoice_Date": "InvoiceDate",
        "Purchase Date": "InvoiceDate",
        "Order Date": "InvoiceDate",
        "Transaction Date": "InvoiceDate",

        # Price
        "Price": "UnitPrice",
        "price": "UnitPrice",
        "Unit Price": "UnitPrice",
        "Unit_Price": "UnitPrice",
        "Purchase Amount": "UnitPrice",
        "Purchase Amount (USD)": "UnitPrice",
        "Amount": "UnitPrice",
        "Sales": "UnitPrice",
        "Revenue": "UnitPrice",

        # Customer
        "Customer ID": "CustomerID",
        "Customer_ID": "CustomerID",
        "customer_id": "CustomerID",
        "User ID": "CustomerID",
        "UserID": "CustomerID",
        "Client ID": "CustomerID",

        # Country / Location
        "country": "Country",
        "Location": "Country",
        "location": "Country",
        "Region": "Country",
        "State": "Country",
        "City": "Country",

        # Category
        "Product Category": "Category",
        "Item Category": "Category",
        "category": "Category",
    }

    df.rename(columns={col: rename_map.get(col, col) for col in df.columns}, inplace=True)

    return df


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fallback columns if the uploaded dataset does not fully match
    the UCI Online Retail format.
    """
    df = df.copy()

    # InvoiceNo fallback
    if "InvoiceNo" not in df.columns:
        df["InvoiceNo"] = ["INV" + str(i + 1) for i in range(len(df))]

    # Description fallback
    if "Description" not in df.columns:
        if "Category" in df.columns:
            df["Description"] = df["Category"].astype(str)
        else:
            df["Description"] = "Unknown Product"

    # StockCode fallback
    if "StockCode" not in df.columns:
        df["StockCode"] = df["Description"].astype(str).factorize()[0] + 1
        df["StockCode"] = "P" + df["StockCode"].astype(str)

    # Quantity fallback
    if "Quantity" not in df.columns:
        df["Quantity"] = 1

    # InvoiceDate fallback
    if "InvoiceDate" not in df.columns:
        df["InvoiceDate"] = pd.date_range(
            start="2024-01-01",
            periods=len(df),
            freq="D"
        )

    # UnitPrice fallback
    if "UnitPrice" not in df.columns:
        df["UnitPrice"] = 0

    # CustomerID fallback
    if "CustomerID" not in df.columns:
        df["CustomerID"] = ["CUST" + str(i + 1) for i in range(len(df))]

    # Country fallback
    if "Country" not in df.columns:
        df["Country"] = "Unknown"

    # Category fallback
    if "Category" not in df.columns:
        df["Category"] = "General"

    return df


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transaction data and add derived analytics columns."""
    df = normalize_columns(df)
    df = add_missing_columns(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". Use the included sample dataset or rename your columns."
        )

    df = df.copy()

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Drop rows where important fields are empty
    df = df.dropna(
        subset=[
            "InvoiceNo",
            "StockCode",
            "Description",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
            "CustomerID",
            "Country",
        ]
    )

    # Convert data types safely
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # Remove invalid converted values
    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])

    # Remove cancelled invoices if present
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove invalid quantity and price
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] >= 0)]

    # Clean CustomerID
    df["CustomerID"] = (
        df["CustomerID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    # Clean text columns
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()

    # Add analytics columns
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["OrderDate"] = df["InvoiceDate"].dt.date
    df["Year"] = df["InvoiceDate"].dt.year
    df["DayName"] = df["InvoiceDate"].dt.day_name()

    return df.reset_index(drop=True)


def save_clean_data(raw_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Load, clean, save and return the processed dataframe."""
    df = load_data(raw_path)
    clean_df = clean_retail_data(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(output_path, index=False)

    return clean_df