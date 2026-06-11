from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import clean_retail_data, load_data
from src.eda import kpi_summary, monthly_revenue, category_performance, revenue_by_country
from src.segmentation import segment_customers
from src.churn_model import train_churn_model
from src.recommendation import top_products
from src.utils import RAW_DATA, PROCESSED_DATA, format_currency, generate_insights

st.set_page_config(
    page_title="AI Consumer Behavior Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_clean_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            raw_df = pd.read_excel(uploaded_file)
        else:
            raw_df = pd.read_csv(uploaded_file)
        return clean_retail_data(raw_df)
    raw_df = load_data(RAW_DATA)
    return clean_retail_data(raw_df)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


load_css()

st.sidebar.title("📊 Consumer Analytics")
st.sidebar.caption("Data Analytics + Machine Learning Project")
uploaded_file = st.sidebar.file_uploader("Upload your CSV/Excel dataset", type=["csv", "xlsx", "xls"])
st.sidebar.info("No dataset? The app automatically uses the included sample retail dataset.")

try:
    df = get_clean_data(uploaded_file)
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_DATA, index=False)

st.markdown('<div class="main-title">AI-Powered Consumer Behavior Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Analyze customers, predict churn, discover segments, and recommend products using AI-driven analytics.</div>',
    unsafe_allow_html=True,
)

kpis = kpi_summary(df)
cols = st.columns(5)
with cols[0]:
    metric_card("Total Revenue", format_currency(kpis["total_revenue"]))
with cols[1]:
    metric_card("Total Orders", f'{kpis["total_orders"]:,}')
with cols[2]:
    metric_card("Customers", f'{kpis["total_customers"]:,}')
with cols[3]:
    metric_card("Avg Order Value", format_currency(kpis["avg_order_value"]))
with cols[4]:
    metric_card("Products", f'{kpis["total_products"]:,}')

st.markdown("---")

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Monthly Revenue Trend")
    revenue_df = monthly_revenue(df)
    fig = px.line(revenue_df, x="Month", y="Revenue", markers=True, title="Revenue by Month")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Category Performance")
    cat_df = category_performance(df)
    fig2 = px.bar(cat_df, x="Category", y="Revenue", title="Revenue by Category")
    fig2.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top Products")
st.dataframe(top_products(df, 10), use_container_width=True, hide_index=True)

st.markdown("---")
seg = segment_customers(df)
churn = train_churn_model(df)
insights = generate_insights(df, seg.rfm, churn.customer_churn)

st.subheader("AI Business Insights")
for insight in insights:
    st.markdown(f"- {insight}")

with st.expander("View Cleaned Dataset"):
    st.dataframe(df.head(100), use_container_width=True)

st.markdown('<div class="footer-note">Built with Python, Pandas, Scikit-learn, Plotly and Streamlit.</div>', unsafe_allow_html=True)
