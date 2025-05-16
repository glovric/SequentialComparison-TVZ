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
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = load_showcase_data(df)

    # Placeholder sample chart (remove or replace with your model output)
    x = np.arange(0, 100)
    y_true = np.sin(x / 10)
    y_pred = y_true + np.random.normal(0, 0.1, size=len(x))

    df_pred = pd.DataFrame({"Time": x, "Actual": y_true, "Predicted": y_pred})
    fig_lstm = px.line(df_pred, x="Time", y=["Actual", "Predicted"], title="Sample LSTM Output (Mock Data)",
                       labels={"value": "Price", "Time": "Time Step"}, template="plotly_white")
    st.plotly_chart(fig_lstm, use_container_width=True)
    #################################################

    scaler = joblib.load('scalers/standard_scaler.save')
    y_true = scaler.inverse_transform(y_train_seq.cpu().detach().numpy())
    y_pred = lstm_model(X_train_seq).cpu().detach().numpy()
    y_pred = scaler.inverse_transform(y_pred)
    line_color_true = '#1f77b4'
    line_color_pred = '#ff7f0e'
    line_cols = st.columns(5)

    for i, feature in enumerate(df.columns):
        col = line_cols[i % 5]
        
        with col:
            idx = df.columns.get_loc(feature)  # get index of the feature in df

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df.index, y=y_true[:, idx],
                mode='lines', name='True', line=dict(color=line_color_true)
            ))

            fig.add_trace(go.Scatter(
                x=df.index, y=y_pred[:, idx],
                mode='lines', name='Predicted', line=dict(color=line_color_pred)
            ))

            fig.update_layout(
                title=f"Line Chart of {feature}",
                template="plotly_white",
                legend=dict(x=0.01, y=0.99),
                margin=dict(t=40, b=20)
            )

            fig.update_xaxes(tickformat="%Y-%m")

            st.plotly_chart(fig, use_container_width=True)

        # Regenerate new columns every 2 plots
        if (i + 1) % 5 == 0:
            line_cols = st.columns(5)