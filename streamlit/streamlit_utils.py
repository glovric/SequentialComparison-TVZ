import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
from utils.torch_utils import load_custom_test_data, LSTMModel, GRUModel, TransformerModel

# Function to calculate the selected metric
def calculate_metric(metric_name, y_true, y_pred, idx):
    if metric_name == "MSE":
        return mean_squared_error(y_true[:, idx], y_pred[:, idx])
    elif metric_name == "RMSE":
        return root_mean_squared_error(y_true[:, idx], y_pred[:, idx])
    elif metric_name == "MAE":
        return mean_absolute_error(y_true[:, idx], y_pred[:, idx])
    elif metric_name == "MAPE":
        return mean_absolute_percentage_error(y_true[:, idx], y_pred[:, idx]) * 100

def display_metric_html(metrics: list[str], metric_results: dict[str, float]):
    return "<div style='text-align: center;'>" + \
                " | ".join([
                f"<strong>{metric}</strong>: {metric_results[metric]:.4f}%" if metric == "MAPE"
                else f"<strong>{metric}</strong>: {metric_results[metric]:.4f}"
                for metric in metrics
                ]) + \
            "</div>"

def render_histograms(df: pd.DataFrame, basic_features: list[str], derived_features: list[str]) -> None:

    # Create 3 columns, two for multiselects and one for color picker
    col1, col2, col3 = st.columns([2, 2, 3])

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

    with col3:
        st.empty()

    # Combine all selected features
    selected_hist_features = basic_selected + derived_selected

    # If any features are selected render the histograms
    if selected_hist_features:

        # Create one column for slider, other will be empty
        col1, col2, col3, col4 = st.columns([2, 2, 2, 3])

        # Use column 1 for slider
        with col1:
            num_columns = st.number_input("Number of columns", min_value=1, max_value=5, step=1, value=2, key="num_columns_hist")
        with col2:
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=30, step=5)
        with col3:
            hist_color = st.color_picker("Pick a histogram color", "#636EFA")
        with col4:
            st.empty()

        # Plot histograms in two columns
        hist_cols = st.columns(num_columns)
        for i, feature in enumerate(selected_hist_features):
            col = hist_cols[i % num_columns]
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
            if (i + 1) % num_columns == 0:
                hist_cols = st.columns(num_columns)

def render_box_whiskers(df: pd.DataFrame, basic_features: list[str], derived_features: list[str]) -> None:
    col1, col2, col3 = st.columns([2, 2, 3])

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
    with col3:
        st.empty()

    selected_box_features = basic_selected_box + derived_selected_box

    if selected_box_features:

        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            num_columns = st.number_input("Number of columns", min_value=1, max_value=5, step=1, value=2, key="num_columns_box")
        with col2:
            box_color = st.color_picker("Pick a box plot color", "#EF553B")
        with col3:
            st.empty()

        box_cols = st.columns(num_columns)
        for i, feature in enumerate(selected_box_features):
            col = box_cols[i % num_columns]
            with col:
                fig = px.box(
                    df, y=feature, title=f"Box Plot of {feature}",
                    template="plotly_white", color_discrete_sequence=[box_color], points="outliers"
                )
                st.plotly_chart(fig, use_container_width=True)
            if (i + 1) % num_columns == 0:
                box_cols = st.columns(num_columns)

def render_line(df: pd.DataFrame, basic_features: list[str], derived_features: list[str]) -> None:
    col1, col2, col3 = st.columns([2, 2, 3])

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

        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            num_columns = st.number_input("Number of columns", min_value=1, max_value=5, step=1, value=2, key="num_columns_line")
        with col2:
            line_color = st.color_picker("Pick a line plot color", "#EF553B")
        with col3:
            st.empty()

        line_cols = st.columns(num_columns)
        for i, feature in enumerate(selected_line_features):
            col = line_cols[i % num_columns]
            with col:
                fig = px.line(
                    df, x=df.index, y=feature, title=f"Line Chart of {feature}",
                    template="plotly_white", color_discrete_sequence=[line_color]
                )
                st.plotly_chart(fig, use_container_width=True)
            if (i + 1) % num_columns == 0:
                line_cols = st.columns(num_columns)

def render_candlestick(df: pd.DataFrame) -> None:
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

def render_data_summary(df: pd.DataFrame) -> None:
    with st.expander("Show raw data"):
        combined = pd.concat([df.head(), df.tail()])
        st.write(combined)
    st.write("📐 DataFrame shape:", df.shape)
    st.write("📊 Descriptive statistics:")
    st.write(df.describe())

def render_model_train(df: pd.DataFrame, model: LSTMModel | GRUModel | TransformerModel, X_train_seq: torch.Tensor, y_train_seq: torch.Tensor, basic_features: list[str], derived_features: list[str]) -> None:

    scaler = joblib.load('scalers/standard_scaler.save')

    if isinstance(model, TransformerModel):
        y_pred = model(X_train_seq).squeeze(1).cpu().detach().numpy()
    else:
        y_pred = model(X_train_seq).cpu().detach().numpy()

    y_true = scaler.inverse_transform(y_train_seq.cpu().detach().numpy())
    y_pred = scaler.inverse_transform(y_pred)

    train_dates = df.index[:len(X_train_seq)]

    col1, col2, col3 = st.columns([2, 2, 2])

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

    with col3:
        metrics = st.multiselect(
            "Select evaluation metrics",
            ["MSE", "RMSE", "MAE", "MAPE"],
            default=["MSE"],  # Default to MSE if no selection is made
            key="metrics_train"
        )

    selected_line_features = basic_selected_line + derived_selected_line  

    if selected_line_features:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

        with col1:
            num_columns = st.number_input("Number of columns", min_value=1, max_value=5, step=1, value=2, key="num_columns_train")
        with col2:
            line_color_true = st.color_picker("True line color", "#1f77b4", key="line_color_true_key_train")
        with col3:
            line_color_pred = st.color_picker("Predicted line color", "#ff7f0e", key="line_color_pred_key_train")
        with col4:
            st.empty()

        st.markdown("<br>", unsafe_allow_html=True)

        line_cols = st.columns(num_columns)

        for i, feature in enumerate(selected_line_features):
            col = line_cols[i % num_columns]
            with col:
                idx = df.columns.get_loc(feature)  # get index of the feature in df

                fig = go.Figure()

                # Add true values trace
                fig.add_trace(go.Scatter(
                    x=train_dates, y=y_true[:, idx],
                    mode='lines', name='True', line=dict(color=line_color_true)
                ))

                # Add predicted values trace
                fig.add_trace(go.Scatter(
                    x=train_dates, y=y_pred[:, idx],
                    mode='lines', name='Predicted', line=dict(color=line_color_pred)
                ))

                # Update the plot title with the selected metrics
                fig.update_layout(
                    title=f"Train results for {feature}",
                    template="plotly_white",
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(t=40, b=20)
                )

                fig.update_xaxes(tickformat="%Y-%m")

                # Render the plot
                st.plotly_chart(fig, use_container_width=True)

                metric_results = {metric : calculate_metric(metric, y_true, y_pred, idx) for metric in metrics}
                metric_display = display_metric_html(metrics, metric_results)
                st.markdown(metric_display, unsafe_allow_html=True)

            if (i + 1) % num_columns == 0:
                line_cols = st.columns(num_columns)

        st.markdown("<hr style='border: 1px solid #bbb; margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)

def render_model_test(df: pd.DataFrame, model: LSTMModel | GRUModel | TransformerModel, X_test_scaled: np.ndarray, basic_features: list[str], derived_features: list[str]) -> None:
    
    scaler = joblib.load('scalers/standard_scaler.save')

    col1, col2, col3 = st.columns([2, 2, 2])

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
    with col3:
        metrics = st.multiselect(
            "Select evaluation metrics",
            ["MSE", "RMSE", "MAE", "MAPE"],
            default=["MSE"],  # Default to MSE if no selection is made,
            key="metrics_test"
        )

    selected_line_features = basic_selected_line + derived_selected_line

    if selected_line_features:

        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 3])

        with col1:
            num_columns = st.number_input("Number of columns", min_value=1, max_value=5, step=1, value=2, key="num_columns_test")
        with col2:
            num_int = st.number_input("Sequence length:", min_value=1, max_value=100, step=1, value=10, key="num_int")
        with col3:
            line_color_true = st.color_picker("True line color", "#1f77b4", key="line_color_true_key_test")
        with col4:
            line_color_pred = st.color_picker("Predicted line color", "#ff7f0e", key="line_color_pred_key_test")
        with col5:
            st.empty()

        st.markdown("<br>", unsafe_allow_html=True)

        X_test_seq, y_test_seq = load_custom_test_data(X_test_scaled, input_seq_len=num_int)

        if isinstance(model, TransformerModel):
            y_pred = model(X_test_seq).squeeze(1).cpu().detach().numpy()
        else:
            y_pred = model(X_test_seq).cpu().detach().numpy()

        y_true = scaler.inverse_transform(y_test_seq.cpu().detach().numpy())
        y_pred = scaler.inverse_transform(y_pred)
        test_dates = df.index[-len(X_test_seq):]

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

                metric_results = {metric : calculate_metric(metric, y_true, y_pred, idx) for metric in metrics}
                metric_display = display_metric_html(metrics, metric_results)
                st.markdown(metric_display, unsafe_allow_html=True)

            if (i + 1) % num_columns == 0:
                line_cols = st.columns(num_columns)

        st.markdown("<hr style='border: 1px solid #bbb; margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)

def render_model_train_multiple(df: pd.DataFrame, model: LSTMModel | GRUModel | TransformerModel, X_train_seq: torch.Tensor, y_train_seq: torch.Tensor, basic_features: list[str], derived_features: list[str]) -> None:
    scaler = joblib.load('scalers/standard_scaler.save')
    col1, col2, col3 = st.columns([2, 2, 2])
    
    # Ensure rand_idx persists
    if 'rand_idx_train' not in st.session_state:
        st.session_state.rand_idx_train = np.random.randint(0, len(X_train_seq))

    rand_idx = st.session_state.rand_idx_train

    x = X_train_seq[rand_idx].cpu().detach().numpy()
    y_true = y_train_seq[rand_idx].cpu().detach().numpy()

    if isinstance(model, TransformerModel):
        y_pred = model(X_train_seq[rand_idx].unsqueeze(1)).squeeze(1).cpu().detach().numpy()
    else:
        y_pred = model(X_train_seq[rand_idx]).cpu().detach().numpy()

    x = scaler.inverse_transform(x)
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)

    with col1:
        basic_selected_line = st.multiselect(
            "Select basic features to plot true vs. predicted",
            basic_features,
            default=["Low"],
            key="basic_features_train_multiple"
        )

    with col2:
        derived_selected_line = st.multiselect(
            "Select derived features to plot true vs. predicted",
            derived_features,
            default=["Daily Return"],
            key="derived_features_train_multiple"            
        )
    with col3:
        metrics = st.multiselect(
                    "Select evaluation metrics",
                    ["MSE", "RMSE", "MAE", "MAPE"],
                    default=["MSE"],  # Default to MSE if no selection is made,
                    key="metrics_train_multiple"
                )

    selected_line_features = basic_selected_line + derived_selected_line

    if selected_line_features:

        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

        with col1:
            num_columns = st.number_input("Number of columns", min_value=1, max_value=5, step=1, value=2, key="num_columns_train_multiple")
        with col2:
            line_color_true = st.color_picker("True line color", "#1FB44F", key="line_color_true_key_train_multiple")
        with col3:
            line_color_pred = st.color_picker("Predicted line color", "#ff7f0e", key="line_color_pred_key_train_multiple")
        with col4:
            st.empty()

        if st.button("Load new random train sequence", key="button_rand_train"):
            st.session_state.rand_idx_train = np.random.randint(0, len(X_train_seq))

        st.markdown("<br>", unsafe_allow_html=True)

        line_cols = st.columns(num_columns)

        for i, feature in enumerate(selected_line_features):
            col = line_cols[i % num_columns]

            with col:
                idx = df.columns.get_loc(feature)  # Get index of feature

                input_seq = x[:, idx]
                true_seq = y_true[:, idx]
                pred_seq = y_pred[:, idx]

                # Construct x-axis ranges for each segment
                input_range = list(range(len(input_seq) + 1))
                true_range = list(range(len(input_seq), len(input_seq) + len(true_seq)))
                pred_range = list(range(len(input_seq), len(input_seq) + len(pred_seq)))

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=input_range,
                    y=np.hstack((input_seq, true_seq[0])),
                    mode='lines',
                    name='Input',
                    line=dict(color='#1f77b4')
                ))

                fig.add_trace(go.Scatter(
                    x=true_range,
                    y=true_seq,
                    mode='lines',
                    name='True',
                    line=dict(color=line_color_true)
                ))

                fig.add_trace(go.Scatter(
                    x=pred_range,
                    y=pred_seq,
                    mode='lines',
                    name='Predicted',
                    line=dict(color=line_color_pred, dash='dash')
                ))

                fig.update_layout(
                    title=f"Train results for {feature}",
                    template="plotly_white",
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(t=40, b=20),
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

                metric_results = {metric : calculate_metric(metric, y_true, y_pred, idx) for metric in metrics}
                metric_display = display_metric_html(metrics, metric_results)
                st.markdown(metric_display, unsafe_allow_html=True)

            if (i + 1) % num_columns == 0:
                line_cols = st.columns(num_columns)

        st.markdown("<hr style='border: 1px solid #bbb; margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)

def render_model_test_multiple(df: pd.DataFrame, model: LSTMModel | GRUModel | TransformerModel, X_test_scaled: np.ndarray, basic_features: list[str], derived_features: list[str]) -> None:
    
    scaler = joblib.load('scalers/standard_scaler.save')
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        basic_selected_line = st.multiselect(
            "Select basic features to plot true vs. predicted",
            basic_features,
            default=["Low"],
            key="basic_features_test_multiple"
        )

    with col2:
        derived_selected_line = st.multiselect(
            "Select derived features to plot true vs. predicted",
            derived_features,
            default=["Daily Return"],
            key="derived_features_test_multiple"
        )
    
    with col3:
        metrics = st.multiselect(
                    "Select evaluation metrics",
                    ["MSE", "RMSE", "MAE", "MAPE"],
                    default=["MSE"],  # Default to MSE if no selection is made,
                    key="metrics_test_multiple"
                )

    selected_line_features = basic_selected_line + derived_selected_line

    if selected_line_features:

        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 3])

        with col1:
            num_columns = st.number_input("Number of columns", min_value=1, max_value=5, step=1, value=2, key="num_columns_test_multiple")
        with col2:
            num_int = st.number_input("Sequence length:", min_value=1, max_value=100, step=1, value=10, key="num_int_multiple")
        with col3:
            line_color_pred = st.color_picker("Predicted line color", "#ff7f0e", key="line_color_pred_key_test_multiple")
        with col4:
            line_color_true = st.color_picker("True line color", "#1FB44F", key="line_color_true_key_test_multiple")
        with col5:
            st.empty()
            
        X_test_seq, y_test_seq = load_custom_test_data(X_test_scaled, input_seq_len=num_int, target_seq_len=num_int)

        # Ensure rand_idx persists
        if 'rand_idx_test' not in st.session_state:
            st.session_state.rand_idx_test = 0

        if st.session_state.rand_idx_test >= len(X_test_seq):
            st.session_state.rand_idx_test = np.random.randint(0, len(X_test_seq)) if len(X_test_seq) > 0 else 0

        rand_idx = st.session_state.rand_idx_test

        x = X_test_seq[rand_idx].cpu().detach().numpy()
        y_true = y_test_seq[rand_idx].cpu().detach().numpy()
        
        if isinstance(model, TransformerModel):
            y_pred = model(X_test_seq[rand_idx].unsqueeze(1)).cpu().detach().numpy()
        else:
            y_pred = model(X_test_seq[rand_idx]).cpu().detach().numpy()

        x = scaler.inverse_transform(x)
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
        

        if st.button("Load new random test sequence", key="button_rand_test"):
            if len(X_test_seq) > 0:
                st.session_state.rand_idx_test = np.random.randint(0, len(X_test_seq))
            else:
                st.session_state.rand_idx_test = 0

            rand_idx = st.session_state.rand_idx_test

        line_cols = st.columns(num_columns)

        for i, feature in enumerate(selected_line_features):
            col = line_cols[i % num_columns]

            with col:
                idx = df.columns.get_loc(feature)  # Get index of feature

                input_seq = x[:, idx]
                true_seq = y_true[:, idx]
                pred_seq = y_pred[:, idx]

                # Construct x-axis ranges for each segment
                input_range = list(range(len(input_seq) + 1))
                true_range = list(range(len(input_seq), len(input_seq) + len(true_seq)))
                pred_range = list(range(len(input_seq), len(input_seq) + len(pred_seq)))

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=input_range,
                    y=np.hstack((input_seq, true_seq[0])),
                    mode='lines',
                    name='Input',
                    line=dict(color='#1f77b4')
                ))

                fig.add_trace(go.Scatter(
                    x=true_range,
                    y=true_seq,
                    mode='lines',
                    name='True',
                    line=dict(color=line_color_true)
                ))

                fig.add_trace(go.Scatter(
                    x=pred_range,
                    y=pred_seq,
                    mode='lines',
                    name='Predicted',
                    line=dict(color=line_color_pred, dash='dash')
                ))

                fig.update_layout(
                    title=f"Test results for {feature}",
                    template="plotly_white",
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(t=40, b=20),
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

                metric_results = {metric : calculate_metric(metric, y_true, y_pred, idx) for metric in metrics}
                metric_display = display_metric_html(metrics, metric_results)
                st.markdown(metric_display, unsafe_allow_html=True)

            if (i + 1) % num_columns == 0:
                line_cols = st.columns(num_columns)