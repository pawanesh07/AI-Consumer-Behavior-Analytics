from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import load_data, clean_retail_data
from src.segmentation import segment_customers
from src.churn_model import train_churn_model
from src.utils import RAW_DATA, PROCESSED_DATA, generate_insights

st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")
st.title("🤖 AI Business Insights")

@st.cache_data(show_spinner=False)
def load_cleaned():
    if PROCESSED_DATA.exists():
        return pd.read_csv(PROCESSED_DATA, parse_dates=["InvoiceDate"])
    return clean_retail_data(load_data(RAW_DATA))

df = load_cleaned()
seg = segment_customers(df)
churn = train_churn_model(df)

st.subheader("Auto-Generated Recommendations")
for i, insight in enumerate(generate_insights(df, seg.rfm, churn.customer_churn), start=1):
    st.success(f"{i}. {insight}")

st.subheader("Action Plan")
st.markdown(
    """
    - Send discount coupons to high churn-risk customers.
    - Create bundles using top-selling products.
    - Focus marketing on high-revenue countries and months.
    - Give loyalty benefits to Champion and Loyal customer segments.
    - Use recommendation output to personalize product suggestions.
    """
)
