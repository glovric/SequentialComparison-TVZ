import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from streamlit_utils import *
from utils import add_financial_features
from torch_utils import *
import joblib

@st.cache_data
def load_data():
    df = yf.download('AAPL')
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df = add_financial_features(df)
    return df

@st.cache_data
def load_model_train_data(df):
    return load_showcase_train_data(df)

st.set_page_config(page_title="Stock Dashboard + LSTM Showcase", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📈 AAPL Stock Dashboard", "🤖 LSTM Model Showcase"])

# --- Load data ---
df = load_data()
df_models = df[df.index >= '2015-01-01']
X_train, y_train, y_train_multiple, X_test_scaled = load_model_train_data(df_models)
available_columns = df.columns.tolist()
basic_features = available_columns[:5]
derived_features = available_columns[5:]


# --- Page 1: AAPL Dashboard ---

if page == "📈 AAPL Stock Dashboard":

    st.title("📊 AAPL Stock Data Dashboard")

    st.subheader("🧮 Data Summary")
    render_data_summary(df)

    st.subheader("Histograms")
    render_histograms(df, basic_features, derived_features)
    
    st.subheader("Box and Whisker Plots")
    render_box_whiskers(df, basic_features, derived_features)

    st.subheader("Line Charts")
    render_line(df, basic_features, derived_features)

    st.subheader("Candlestick Chart")
    render_candlestick(df)


# --- Page 2: LSTM Model Showcase ---

elif page == "🤖 LSTM Model Showcase":

    st.title("🤖 LSTM Model Showcase")

    lstm_model = load_lstm_model()
    lstm_model2 = load_lstm_model(predict_sequence=True)
    scaler = joblib.load('scalers/standard_scaler.save')

    st.subheader("LSTM many-to-one Training Results")
    render_lstm_train(df_models, lstm_model, X_train, y_train, basic_features, derived_features)

    st.subheader("LSTM many-to-one Test Results")
    render_lstm_test(df_models, lstm_model, X_test_scaled, basic_features, derived_features)

    st.subheader("LSTM many-to-many Training Results")
    render_lstm_train2(df_models, lstm_model2, X_train, y_train_multiple, basic_features, derived_features)