import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Dashboard + LSTM Showcase", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📈 AAPL Stock Dashboard", "🤖 LSTM Model Showcase"])

# --- Shared Functions ---
@st.cache_data
def load_data():
    df = yf.download('AAPL')
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    return df

def add_financial_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df["Daily Return"] = (df["Close"] - df["Open"]) / df["Open"]
    close_pct_change = df['Close'].pct_change()
    df['Lagged Return'] = close_pct_change.shift(1).fillna(0)
    df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    df[f"SMA {window}"] = df["Close"].rolling(window=window).mean().fillna(0)
    df["Prev_Close"] = df['Close'].shift(1).fillna(0)
    true_range = df[['High', 'Low', 'Prev_Close']].apply(
        lambda x: max(x["High"] - x["Low"], abs(x["High"] - x["Prev_Close"]), abs(x["Low"] - x["Prev_Close"])), axis=1)
    df[f'ATR {window}'] = true_range.rolling(window=window).mean().fillna(0)
    df.drop(["Prev_Close"], axis=1, inplace=True)
    return df

# --- Page 1: AAPL Dashboard ---

if page == "📈 AAPL Stock Dashboard":

    st.title("📊 AAPL Stock Data Dashboard")
    df = load_data()
    df = add_financial_features(df)
    available_columns = df.columns.tolist()

    # --- Multi-feature Histograms ---
    st.subheader("📊 Multiple Histograms")

    selected_hist_features = st.multiselect(
        "Select features to plot histograms", available_columns, default=["Daily Return"]
    )
    col1, col2 = st.columns(2)
    with col1:
        bins = st.slider("Number of bins", min_value=5, max_value=100, value=30, step=5)
    with col2:
        hist_color = st.color_picker("Pick a histogram color", "#636EFA")

    if selected_hist_features:

        hist_cols = st.columns(2)
        for i, feature in enumerate(selected_hist_features):
            col = hist_cols[i % 2]
            with col:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[feature],
                    nbinsx=bins,
                    marker=dict(
                        color=hist_color,
                        line=dict(color='rgb(0, 0, 0)', width=0.3)
                    ),
                    name=feature
                ))
                fig.update_layout(
                    title=f"Histogram of {feature}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            if (i + 1) % 2 == 0:
                hist_cols = st.columns(2)

    # --- Multi-feature Box and Whisker Plots ---
    st.subheader("📦 Multiple Box and Whisker Plots")

    selected_box_features = st.multiselect(
        "Select features to plot box plots", available_columns, default=["Close"]
    )
    box_color = st.color_picker("Pick a box plot color", "#EF553B")

    if selected_box_features:
        box_cols = st.columns(2)
        for i, feature in enumerate(selected_box_features):
            col = box_cols[i % 2]
            with col:
                fig = px.box(
                    df, y=feature, title=f"Box Plot of {feature}",
                    template="plotly_white", color_discrete_sequence=[box_color], points="outliers"
                )
                st.plotly_chart(fig, use_container_width=True)
            if (i + 1) % 2 == 0:
                box_cols = st.columns(2)

    # --- Multi-feature Line Charts ---
    st.subheader("📈 Multiple Line Charts")

    selected_line_features = st.multiselect(
        "Select features to plot line charts", available_columns, default=["Close", "SMA 14"]
    )

    if selected_line_features:
        line_cols = st.columns(2)
        for i, feature in enumerate(selected_line_features):
            col = line_cols[i % 2]
            with col:
                color = st.color_picker(f"Pick a color for {feature}", "#00CC96", key=f"line_color_{feature}")
                fig = px.line(
                    df, x=df.index, y=feature, title=f"Line Chart of {feature}",
                    template="plotly_white", color_discrete_sequence=[color]
                )
                st.plotly_chart(fig, use_container_width=True)
            if (i + 1) % 2 == 0:
                line_cols = st.columns(2)

    # --- Candlestick Chart ---
    st.subheader("🕯️ Candlestick Chart")

    fig_candle = go.Figure(data=[
        go.Candlestick(x=df.index,
                       open=df['Open'],
                       high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       increasing_line_color='green',
                       decreasing_line_color='red')
    ])

    fig_candle.update_layout(
        title="AAPL Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )

    st.plotly_chart(fig_candle, use_container_width=True)

    # --- Data Summary ---
    st.subheader("🧮 Data Summary")
    with st.expander("Show raw data"):
        st.write(df.tail())
    st.write("📐 DataFrame shape:", df.shape)
    st.write("📊 Descriptive statistics:")
    st.write(df.describe())


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