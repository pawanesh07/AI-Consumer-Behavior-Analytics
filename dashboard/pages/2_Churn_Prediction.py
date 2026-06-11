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
from src.churn_model import train_churn_model
from src.utils import RAW_DATA, PROCESSED_DATA

st.set_page_config(page_title="Churn Prediction", page_icon="⚠️", layout="wide")
st.title("⚠️ Customer Churn Prediction")

@st.cache_data(show_spinner=False)
def load_cleaned():
    if PROCESSED_DATA.exists():
        return pd.read_csv(PROCESSED_DATA, parse_dates=["InvoiceDate"])
    return clean_retail_data(load_data(RAW_DATA))

df = load_cleaned()
result = train_churn_model(df)
churn_df = result.customer_churn.copy()

c1, c2, c3 = st.columns(3)
c1.metric("Model Accuracy", f"{result.accuracy * 100:.2f}%")
c2.metric("High Risk Customers", int((churn_df["RiskLevel"] == "High").sum()))
c3.metric("Medium Risk Customers", int((churn_df["RiskLevel"] == "Medium").sum()))

left, right = st.columns([1, 1])
with left:
    fig = px.histogram(churn_df, x="ChurnProbability", nbins=20, title="Churn Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)
with right:
    risk_counts = churn_df["RiskLevel"].value_counts().reset_index()
    risk_counts.columns = ["RiskLevel", "Customers"]
    fig = px.pie(risk_counts, names="RiskLevel", values="Customers", title="Risk Level Split")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("High-Risk Customer List")
st.dataframe(
    churn_df.sort_values("ChurnProbability", ascending=False).head(25),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Model Classification Report"):
    st.code(result.report)
