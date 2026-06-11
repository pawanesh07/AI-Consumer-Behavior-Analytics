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
from src.segmentation import segment_customers
from src.utils import RAW_DATA, PROCESSED_DATA, format_currency

st.set_page_config(page_title="Segmentation", page_icon="🎯", layout="wide")
st.title("🎯 Customer Segmentation")

@st.cache_data(show_spinner=False)
def load_cleaned():
    if PROCESSED_DATA.exists():
        return pd.read_csv(PROCESSED_DATA, parse_dates=["InvoiceDate"])
    return clean_retail_data(load_data(RAW_DATA))

df = load_cleaned()
result = segment_customers(df)
rfm = result.rfm

st.subheader("RFM-Based Customer Segments")
col1, col2, col3 = st.columns(3)
col1.metric("Average Recency", f"{rfm['Recency'].mean():.1f} days")
col2.metric("Average Frequency", f"{rfm['Frequency'].mean():.1f} orders")
col3.metric("Average Monetary", format_currency(rfm["Monetary"].mean()))

left, right = st.columns(2)
with left:
    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    st.plotly_chart(px.bar(seg_counts, x="Segment", y="Customers", title="Customers by Segment"), use_container_width=True)
with right:
    st.plotly_chart(px.scatter(rfm, x="Recency", y="Monetary", size="Frequency", color="Segment", title="RFM Segment Map"), use_container_width=True)

st.subheader("Customer Segment Table")
st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True, hide_index=True)
