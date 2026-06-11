# AI-Powered Consumer Behavior Analytics Platform

A full working data analytics and machine learning project that analyzes customer purchase behavior, segments customers, predicts churn risk, recommends products, and generates business insights through an interactive Streamlit dashboard.

## Project Features

- Customer analytics dashboard
- Revenue and sales trend analysis
- Product and category performance analysis
- RFM customer segmentation
- K-Means clustering
- Customer churn prediction using Random Forest
- Product recommendation engine
- Auto-generated business insights
- Upload custom CSV/Excel dataset
- Included sample dataset for instant testing

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Machine Learning

## Folder Structure

```text
AI-Consumer-Behavior-Analytics/
├── data/
│   ├── raw/
│   │   └── online_retail_sample.csv
│   └── processed/
├── dashboard/
│   ├── app.py
│   ├── assets/
│   │   └── style.css
│   └── pages/
│       ├── 1_Customer_Analytics.py
│       ├── 2_Churn_Prediction.py
│       ├── 3_Segmentation.py
│       ├── 4_Recommendation.py
│       └── 5_AI_Insights.py
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── segmentation.py
│   ├── churn_model.py
│   ├── recommendation.py
│   └── utils.py
├── models/
├── notebooks/
├── reports/
├── main.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Open project folder in VS Code

```bash
cd AI-Consumer-Behavior-Analytics
code .
```

### 2. Create virtual environment

Windows PowerShell:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run preprocessing pipeline

```bash
python main.py
```

### 5. Run dashboard

```bash
streamlit run dashboard/app.py
```

Then open the URL shown in terminal, usually:

```text
http://localhost:8501
```

## Dataset Format

The app works with the included sample dataset. You can also upload a CSV/Excel file from the dashboard.

Required columns:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

Optional column:

- Category

## Resume Description

Developed an AI-Powered Consumer Behavior Analytics Platform using Python, Pandas, Scikit-learn, Plotly, and Streamlit. The system analyzes customer purchase behavior, performs RFM-based segmentation, predicts customer churn risk using machine learning, recommends products based on similar customer purchases, and generates interactive business insights through a dashboard.

## GitHub Commit Workflow

```bash
git add .
git commit -m "Added consumer behavior analytics dashboard"
git push
```
