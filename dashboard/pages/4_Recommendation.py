from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import load_data, clean_retail_data
from src.recommendation import recommend_for_customer, top_products
from src.utils import RAW_DATA, PROCESSED_DATA

st.set_page_config(page_title="Recommendation", page_icon="🛒", layout="wide")
st.title("🛒 Product Recommendation Engine")

@st.cache_data(show_spinner=False)
def load_cleaned():
    if PROCESSED_DATA.exists():
        return pd.read_csv(PROCESSED_DATA, parse_dates=["InvoiceDate"])
    return clean_retail_data(load_data(RAW_DATA))

df = load_cleaned()
customers = sorted(df["CustomerID"].astype(str).unique())
selected_customer = st.selectbox("Select Customer ID", customers)

st.subheader("Customer Purchase History")
history = df[df["CustomerID"].astype(str) == selected_customer][["InvoiceNo", "InvoiceDate", "Description", "Quantity", "UnitPrice", "TotalAmount"]]
st.dataframe(history.sort_values("InvoiceDate", ascending=False).head(30), use_container_width=True, hide_index=True)

st.subheader("Recommended Products")
recs = recommend_for_customer(df, selected_customer, n=8)
st.dataframe(recs, use_container_width=True, hide_index=True)

st.subheader("Overall Popular Products")
st.dataframe(top_products(df, 10), use_container_width=True, hide_index=True)
