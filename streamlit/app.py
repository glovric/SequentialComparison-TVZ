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
    df = yf.download('AAPL', period='max')
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
    st.markdown("<br>", unsafe_allow_html=True)

    lstm_model.eval()
    lstm_model2.eval()

    st.subheader("LSTM many-to-one Training Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_train(df_models, lstm_model, X_train, y_train, basic_features, derived_features)

    st.subheader("LSTM many-to-one Test Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_test(df_models, lstm_model, X_test_scaled, basic_features, derived_features)

    st.subheader("LSTM many-to-many Training Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_train_multiple(df_models, lstm_model2, X_train, y_train_multiple, basic_features, derived_features)

    st.subheader("LSTM many-to-many Test Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_test_multiple(df_models, lstm_model2, X_test_scaled, basic_features, derived_features)

if page == "🔄 GRU Model Showcase":

    st.title("🔄 GRU Model Showcase")
    st.markdown("<br>", unsafe_allow_html=True)


    gru_model.eval()
    gru_model2.eval()

    st.subheader("GRU many-to-one Training Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_train(df_models, gru_model, X_train, y_train, basic_features, derived_features)

    st.subheader("GRU many-to-one Test Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_test(df_models, gru_model, X_test_scaled, basic_features, derived_features)

    st.subheader("GRU many-to-many Training Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_train_multiple(df_models, gru_model2, X_train, y_train_multiple, basic_features, derived_features)

    st.subheader("GRU many-to-many Test Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_test_multiple(df_models, gru_model2, X_test_scaled, basic_features, derived_features)

if page == "⚡ Transformer Model Showcase":

    st.title("⚡ Transformer Model Showcase")
    st.markdown("<br>", unsafe_allow_html=True)

    transformer_model.eval()
    transformer_model2.eval()

    st.subheader("Transformer many-to-one Training Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_train(df_models, transformer_model, X_train, y_train, basic_features, derived_features)

    st.subheader("Transformer many-to-one Test Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_test(df_models, transformer_model, X_test_scaled, basic_features, derived_features)

    st.subheader("Transformer many-to-many Training Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_train_multiple(df_models, transformer_model2, X_train, y_train_multiple, basic_features, derived_features)

    st.subheader("Transformer many-to-many Test Results")
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_test_multiple(df_models, transformer_model2, X_test_scaled, basic_features, derived_features)

if page == "📊 Model Comparison":

    st.title("📊 Model Comparison")

    st.write("Models were trained using PyTorch CUDA, device NVIDIA GeForce GTX 1650, 4GB VRAM.")
    st.markdown("""
    - Optimizer: Adam  
    - Loss function: MSE
    - Many-to-one number of epochs: 200  
    - Many-to-many number of epochs: 500
    """)

    st.subheader("Hyperparameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### *LSTM*")
        st.markdown("""
        - Hidden dimension: 128  
        - Number of LSTM layers: 3  
        - Dropout: 0.3  
        """)

    with col2:
        st.markdown("##### *GRU*")
        st.markdown("""
        - Hidden dimension: 128  
        - Number of GRU layers: 3  
        - Dropout: 0.3  
        """)

    with col3:
        st.markdown("##### *Transformer*")
        st.markdown("""
        - Number of encoder/decoder layers: 3  
        - Number of Attention Heads: 5  
        - Feed-forward dimension: 2048  
        - Dropout: 0.3  
        """)


    m2o_metrics = {
        "Model": ["LSTM", "GRU", "Transformer"],
        "Training Time (s)": [30.386, 25.116, 75.155], 
        "Train Score (MSE)": [0.1206, 0.0714, 0.1066],
        "Test Score (MSE)": [0.7709, 0.7002, 0.622]
    }

    m2m_metrics = {
        "Model": ["LSTM", "GRU", "Transformer"],
        "Training Time (s)": [57.7, 44.5, 196.3], 
        "Train Score (MSE)": [0.0469, 0.0882, 0.0451],
        "Test Score (MSE)": [0.93975, 1.0225, 0.5546]
    }

    df_m2o = pd.DataFrame(m2o_metrics)
    df_m2m = pd.DataFrame(m2m_metrics)
    df_m2o.index = range(1, len(df_m2o) + 1)
    df_m2m.index = range(1, len(df_m2m) + 1)

    st.subheader("Many-to-one Performance Metrics")
    st.table(df_m2o)
    st.subheader("Many-to-many Performance Metrics")
    st.table(df_m2m)