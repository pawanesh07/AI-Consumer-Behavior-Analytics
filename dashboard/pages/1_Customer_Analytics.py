from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import load_data, clean_retail_data
from src.eda import category_performance, monthly_revenue, revenue_by_country
from src.recommendation import top_products
from src.utils import RAW_DATA, PROCESSED_DATA, format_currency

st.set_page_config(page_title="Customer Analytics", page_icon="👥", layout="wide")
st.title("👥 Customer Analytics")

@st.cache_data(show_spinner=False)
def load_cleaned():
    if PROCESSED_DATA.exists():
        return pd.read_csv(PROCESSED_DATA, parse_dates=["InvoiceDate"])
    return clean_retail_data(load_data(RAW_DATA))

df = load_cleaned()

st.subheader("Customer Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{df['CustomerID'].nunique():,}")
col2.metric("Orders", f"{df['InvoiceNo'].nunique():,}")
col3.metric("Revenue", format_currency(df["TotalAmount"].sum()))
col4.metric("Average Quantity", f"{df['Quantity'].mean():.2f}")

left, right = st.columns(2)
with left:
    country = revenue_by_country(df)
    fig = px.bar(country, x="Country", y="TotalAmount", title="Revenue by Country")
    st.plotly_chart(fig, use_container_width=True)

with right:
    category = category_performance(df)
    fig = px.pie(category, names="Category", values="Revenue", title="Revenue Share by Category")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Monthly Revenue")
st.plotly_chart(px.line(monthly_revenue(df), x="Month", y="Revenue", markers=True), use_container_width=True)

st.subheader("Top Products")
st.dataframe(top_products(df, 15), use_container_width=True, hide_index=True)
