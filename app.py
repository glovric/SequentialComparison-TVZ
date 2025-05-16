import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from streamlit_utils import *
from utils import add_financial_features, get_financial_data
from torch_utils import load_lstm_model, load_showcase_data
import joblib

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
    """)

    df = get_financial_data()
    lstm_model = load_lstm_model()
    scaler = joblib.load('scalers/standard_scaler.save')
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = load_showcase_data(df)

    render_lstm_train(df, lstm_model, X_train_seq, y_train_seq)

    render_lstm_test(df, lstm_model, X_test_seq, y_test_seq)

    scaler = joblib.load('scalers/standard_scaler.save')
    y_true = scaler.inverse_transform(y_test_seq.cpu().detach().numpy())
    y_pred = lstm_model(X_test_seq).cpu().detach().numpy()
    y_pred = scaler.inverse_transform(y_pred)

    for idx, c in enumerate(df.columns[-3:], start=7):
        df_pred = pd.DataFrame({"Time": df.index[-len(y_true):], "Actual": y_true[:, idx], "Predicted": y_pred[:, idx]})
        fig_lstm = px.line(df_pred, x="Time", y=["Actual", "Predicted"], title=f"Test result for {c}",
                        labels={"value": "Price", "Time": "Time Step"}, template="plotly_white")
        st.plotly_chart(fig_lstm, use_container_width=True)