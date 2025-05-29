import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import yfinance as yf
from streamlit_utils import *
from utils.finance_utils import add_financial_features
from utils.torch_utils import *

@st.cache_data
def load_data():
    df = yf.download('AAPL')
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df = add_financial_features(df)
    return df

@st.cache_data
def load_model_train_data(df):
    return load_showcase_train_data(df)

@st.cache_data
def load_models():
    lstm_model = load_lstm_model()
    lstm_model2 = load_lstm_model(predict_sequence=True)
    gru_model = load_gru_model()
    gru_model2 = load_gru_model(predict_sequence=True)
    transformer_model = load_transformer_model()
    transformer_model2 = load_transformer_model(predict_sequence=True)
    return lstm_model, lstm_model2, gru_model, gru_model2, transformer_model, transformer_model2

# Config
st.set_page_config(page_title="Stock Dashboard + Models Showcase", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📈 AAPL Stock Dashboard",
    "🤖 LSTM Model Showcase",
    "🔄 GRU Model Showcase",
    "⚡ Transformer Model Showcase",
    "📊 Model Comparison"
])

df = load_data()
df_models = df[df.index >= '2015-01-01'] # Models were trained with this data
X_train, y_train, y_train_multiple, X_test_scaled = load_model_train_data(df_models)
available_columns = df.columns.tolist()
basic_features = available_columns[:5]
derived_features = available_columns[5:]

lstm_model, lstm_model2, gru_model, gru_model2, transformer_model, transformer_model2 = load_models()

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

elif page == "🤖 LSTM Model Showcase":

    st.title("🤖 LSTM Model Showcase")

    lstm_model.eval()
    lstm_model2.eval()

    st.subheader("LSTM many-to-one Training Results")
    render_lstm_train(df_models, lstm_model, X_train, y_train, basic_features, derived_features)

    st.subheader("LSTM many-to-one Test Results")
    render_lstm_test(df_models, lstm_model, X_test_scaled, basic_features, derived_features)

    st.subheader("LSTM many-to-many Training Results")
    render_lstm_train2(df_models, lstm_model2, X_train, y_train_multiple, basic_features, derived_features)

    st.subheader("LSTM many-to-many Test Results")
    render_lstm_test2(df_models, lstm_model2, X_test_scaled, basic_features, derived_features)

if page == "🔄 GRU Model Showcase":

    st.title("🔄 GRU Model Showcase")

    gru_model.eval()
    gru_model2.eval()

    st.subheader("GRU many-to-one Training Results")
    render_lstm_train(df_models, gru_model, X_train, y_train, basic_features, derived_features)

    st.subheader("GRU many-to-one Test Results")
    render_lstm_test(df_models, gru_model, X_test_scaled, basic_features, derived_features)

    st.subheader("GRU many-to-many Training Results")
    render_lstm_train2(df_models, gru_model2, X_train, y_train_multiple, basic_features, derived_features)

    st.subheader("GRU many-to-many Test Results")
    render_lstm_test2(df_models, gru_model2, X_test_scaled, basic_features, derived_features)

if page == "⚡ Transformer Model Showcase":

    st.title("⚡ Transformer Model Showcase")

    transformer_model.eval()
    transformer_model2.eval()

    st.subheader("Transformer many-to-one Training Results")
    render_lstm_train(df_models, transformer_model, X_train, y_train, basic_features, derived_features)

    st.subheader("Transformer many-to-one Test Results")
    render_lstm_test(df_models, transformer_model, X_test_scaled, basic_features, derived_features)

    st.subheader("Transformer many-to-many Training Results")
    render_lstm_train2(df_models, transformer_model2, X_train, y_train_multiple, basic_features, derived_features)

    st.subheader("Transformer many-to-many Test Results")
    render_lstm_test2(df_models, transformer_model2, X_test_scaled, basic_features, derived_features)

if page == "📊 Model Comparison":

    st.title("📊 Model Comparison")

    # Sample model metrics
    comparison_data = {
        "Model": ["LSTM", "GRU", "Transformer"],
        "Training Time (s)": [30.386, 25.116, 75.155],
        "Train Score (MSE)": [0.1206, 0.0714, 0.1066],
        "Test Score (MSE)": [0.7709, 0.7002, 0.622]
    }

    df_comparison = pd.DataFrame(comparison_data)

    st.subheader("Performance Metrics")
    st.table(df_comparison)