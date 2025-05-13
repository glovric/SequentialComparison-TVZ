import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from streamlit_utils import *
from utils import add_financial_features

@st.cache_data
def load_data():
    df = yf.download('AAPL')
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df = add_financial_features(df)
    return df

st.set_page_config(page_title="Stock Dashboard + LSTM Showcase", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📈 AAPL Stock Dashboard", "🤖 LSTM Model Showcase"])

# --- Page 1: AAPL Dashboard ---

if page == "📈 AAPL Stock Dashboard":

    st.title("📊 AAPL Stock Data Dashboard")

    df = load_data()

    available_columns = df.columns.tolist()
    basic_features = available_columns[:5]
    derived_features = available_columns[5:]

    # --- Data Summary ---
    st.subheader("🧮 Data Summary")
    render_data_summary(df)

    st.subheader("Histograms")
    render_histograms(df, basic_features, derived_features)
    
    st.subheader("Box and Whisker Plots")
    render_box_whiskers(df, basic_features, derived_features)

    # --- Multi-feature Line Charts ---
    st.subheader("Line Charts")
    render_line(df, basic_features, derived_features)

    # --- Candlestick Chart ---
    st.subheader("Candlestick Chart")
    render_candlestick(df)


# --- Page 2: LSTM Model Showcase ---

elif page == "🤖 LSTM Model Showcase":

    st.title("🤖 LSTM Model Showcase")
    st.markdown("""
        This tab is reserved for displaying predictions, evaluation metrics, and charts from your LSTM time series model.
                
        🧠 You can integrate:
        - Prediction vs Actual plots
        - RMSE, MAE, R² metrics
        - Interactive input (e.g., forecast next N days)
    """)

    # Placeholder sample chart (remove or replace with your model output)
    x = np.arange(0, 100)
    y_true = np.sin(x / 10)
    y_pred = y_true + np.random.normal(0, 0.1, size=len(x))

    df_pred = pd.DataFrame({"Time": x, "Actual": y_true, "Predicted": y_pred})
    fig_lstm = px.line(df_pred, x="Time", y=["Actual", "Predicted"], title="Sample LSTM Output (Mock Data)",
                       labels={"value": "Price", "Time": "Time Step"}, template="plotly_white")
    st.plotly_chart(fig_lstm, use_container_width=True)