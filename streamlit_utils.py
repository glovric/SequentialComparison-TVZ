import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from torch_utils import load_custom_test_data
import joblib
import pandas as pd

def render_histograms(df, basic_features, derived_features):

    # Create 3 columns, two for multiselects and one for color picker
    col1, col2, col3 = st.columns([2, 2, 1])

    # Use column 1 for basic features
    with col1:
        basic_selected = st.multiselect(
            "Select basic features to plot histograms",
            basic_features,
            default=["Low"]
        )

    # Use column 2 for derived features
    with col2:
        derived_selected = st.multiselect(
            "Select derived features to plot histograms",
            derived_features,
            default=["Daily Return"]
        )

    # Combine all selected features
    selected_hist_features = basic_selected + derived_selected

    # If any features are selected render the histograms
    if selected_hist_features:

        # Show color picker in third column
        with col3:
            hist_color = st.color_picker("Pick a histogram color", "#636EFA")

        # Create one column for slider, other will be empty
        col1, col2 = st.columns([1, 3])

        # Use column 1 for slider
        with col1:
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=30, step=5)
        with col2:
            st.empty()

        # Plot histograms in two columns
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

def render_box_whiskers(df, basic_features, derived_features):
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        basic_selected_box = st.multiselect(
            "Select basic features to plot box plots",
            basic_features,
            default=["Low"]
        )

    with col2:
        derived_selected_box = st.multiselect(
            "Select derived features to plot box plots",
            derived_features,
            default=["Daily Return"]
        )

    selected_box_features = basic_selected_box + derived_selected_box

    if selected_box_features:
        with col3:
            box_color = st.color_picker("Pick a box plot color", "#EF553B")
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

def render_line(df, basic_features, derived_features):
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        basic_selected_line = st.multiselect(
            "Select basic features to plot line plots",
            basic_features,
            default=["Low"]
        )

    with col2:
        derived_selected_line = st.multiselect(
            "Select derived features to plot line plots",
            derived_features,
            default=["Daily Return"]
        )

    selected_line_features = basic_selected_line + derived_selected_line

    if selected_line_features:
        with col3:
            line_color = st.color_picker("Pick a line plot color", "#EF553B")
        line_cols = st.columns(2)
        for i, feature in enumerate(selected_line_features):
            col = line_cols[i % 2]
            with col:
                fig = px.line(
                    df, x=df.index, y=feature, title=f"Line Chart of {feature}",
                    template="plotly_white", color_discrete_sequence=[line_color]
                )
                st.plotly_chart(fig, use_container_width=True)
            if (i + 1) % 2 == 0:
                line_cols = st.columns(2)

def render_candlestick(df):
    show_candlestick = st.checkbox("Show Candlestick Chart", value=True)

    if show_candlestick:

        fig_candle = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )
        ])

        fig_candle.update_layout(
            title="AAPL Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )

        st.plotly_chart(fig_candle, use_container_width=True)

def render_data_summary(df):
    with st.expander("Show raw data"):
        st.write(df.head())
    st.write("📐 DataFrame shape:", df.shape)
    st.write("📊 Descriptive statistics:")
    st.write(df.describe())

def render_lstm_train(df, model, X_train_seq, y_train_seq, basic_features, derived_features):

    scaler = joblib.load('scalers/standard_scaler.save')
    y_true = scaler.inverse_transform(y_train_seq.cpu().detach().numpy())
    y_pred = model(X_train_seq).cpu().detach().numpy()
    y_pred = scaler.inverse_transform(y_pred)

    train_dates = df.index[:len(X_train_seq)]

    col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])

    with col1:
        basic_selected_line = st.multiselect(
            "Select basic features to plot true vs. predicted",
            basic_features,
            default=["Low"],
            key="basic_features_train"
        )

    with col2:
        derived_selected_line = st.multiselect(
            "Select derived features to plot true vs. predicted",
            derived_features,
            default=["Daily Return"],
            key="derived_features_train"            
        )

    selected_line_features = basic_selected_line + derived_selected_line

    if selected_line_features:

        with col3:
            num_columns = st.number_input("Number of columns", min_value=1, max_value=5, step=1, value=2, key="num_columns_train")
        with col4:
            line_color_true = st.color_picker("True line color", "#1f77b4", key="line_color_true_key_train")
        with col5:
            line_color_pred = st.color_picker("Predicted line color", "#ff7f0e", key="line_color_pred_key_train")

        line_cols = st.columns(num_columns)

        for i, feature in enumerate(selected_line_features):

            col = line_cols[i % num_columns]

            with col:
                idx = df.columns.get_loc(feature)  # get index of the feature in df

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=train_dates, y=y_true[:, idx],
                    mode='lines', name='True', line=dict(color=line_color_true)
                ))

                fig.add_trace(go.Scatter(
                    x=train_dates, y=y_pred[:, idx],
                    mode='lines', name='Predicted', line=dict(color=line_color_pred)
                ))

                fig.update_layout(
                    title=f"Train results for {feature}",
                    template="plotly_white",
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(t=40, b=20)
                )

                fig.update_xaxes(tickformat="%Y-%m")

                st.plotly_chart(fig, use_container_width=True)

            if (i + 1) % num_columns == 0:
                line_cols = st.columns(num_columns)

def render_lstm_test(df, model, X_test_scaled, basic_features, derived_features):
    
    scaler = joblib.load('scalers/standard_scaler.save')

    col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 1, 1, 1])

    with col1:
        basic_selected_line = st.multiselect(
            "Select basic features to plot true vs. predicted",
            basic_features,
            default=["Low"],
            key="basic_features_test"
        )

    with col2:
        derived_selected_line = st.multiselect(
            "Select derived features to plot true vs. predicted",
            derived_features,
            default=["Daily Return"],
            key="derived_features_test"
        )
    selected_line_features = basic_selected_line + derived_selected_line

    if selected_line_features:

        with col3:
            num_columns = st.number_input("Number of columns", min_value=1, max_value=5, step=1, value=2, key="num_columns_test")
        with col4:
            line_color_true = st.color_picker("True line color", "#1f77b4", key="line_color_true_key_test")
        with col5:
            line_color_pred = st.color_picker("Predicted line color", "#ff7f0e", key="line_color_pred_key_test")
        with col6:
            num_int = st.number_input("Sequence length:", min_value=1, max_value=100, step=1, value=10)
            X_test_seq, y_test_seq = load_custom_test_data(X_test_scaled, seq_len=num_int)
            y_true = scaler.inverse_transform(y_test_seq.cpu().detach().numpy())
            y_pred = model(X_test_seq).cpu().detach().numpy()
            y_pred = scaler.inverse_transform(y_pred)
            test_dates = df.index[-len(X_test_seq):]

        st.write(f"Showing test results for sequence length {num_int}, data shape: {tuple(X_test_seq.shape)}")


        line_cols = st.columns(num_columns)

        for i, feature in enumerate(selected_line_features):

            col = line_cols[i % num_columns]

            with col:
                idx = df.columns.get_loc(feature)

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=test_dates, y=y_true[:, idx],
                    mode='lines', name='True', line=dict(color=line_color_true)
                ))

                fig.add_trace(go.Scatter(
                    x=test_dates, y=y_pred[:, idx],
                    mode='lines', name='Predicted', line=dict(color=line_color_pred)
                ))

                fig.update_layout(
                    title=f"Test results for {feature}",
                    template="plotly_white",
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(t=40, b=20)
                )

                fig.update_xaxes(tickformat="%Y-%m")

                st.plotly_chart(fig, use_container_width=True)

            if (i + 1) % num_columns == 0:
                line_cols = st.columns(num_columns)

def render_lstm_test2(df, y_true, y_pred):
    for idx, c in enumerate(df.columns[:5]):
        df_pred = pd.DataFrame({"Time": df.index[-len(y_true):], "Actual": y_true[:, idx], "Predicted": y_pred[:, idx]})
        fig_lstm = px.line(df_pred, x="Time", y=["Actual", "Predicted"], title=f"Test result for {c}",
                        labels={"value": "Price", "Time": "Time Step"}, template="plotly_white")
        st.plotly_chart(fig_lstm, use_container_width=True)

    """for idx, c in enumerate(df.columns[5:], start=5):
        df_pred = pd.DataFrame({"Time": df.index[-len(y_true):], "Actual": y_true[:, idx], "Predicted": y_pred[:, idx]})
        fig_lstm = px.line(df_pred, x="Time", y=["Actual", "Predicted"], title=f"Test result for {c}",
                        labels={"value": "Price", "Time": "Time Step"}, template="plotly_white")
        st.plotly_chart(fig_lstm, use_container_width=True)"""