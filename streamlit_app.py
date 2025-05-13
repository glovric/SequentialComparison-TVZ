# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yfinance as yf
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def random_color():
    return (np.random.random(), np.random.random(), np.random.random())

# Set the title of the app
st.title("Data Science Streamlit Template")

# Sidebar for user interaction
st.sidebar.header("User Input")

tickers = ['', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']

# Create a dropdown menu in the sidebar
ticker = st.sidebar.selectbox("Select a ticker", tickers)

if ticker != '':
    # Read the uploaded CSV file
    df = yf.download(tickers=ticker)
    df.columns = df.columns.droplevel(1)
    
    # Display basic data information
    st.header("Dataset Overview")
    st.write("Shape of the data:", df.shape)
    st.write("Columns in the dataset:", df.columns)

    # Show first few rows of the dataset
    st.subheader("Preview of Data")
    st.write(df.head())

    # Display summary statistics of the data
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Data visualization options
    st.subheader("Visualizations")

    # Histogram of a selected column
    # Interactive Histograms Grid
    st.subheader("Interactive Histograms for Each Column")
    
    # Create a subplot grid for histograms
    rows = 3
    cols = 2
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=df.columns)

    # Iterate through the dataframe columns and create histograms
    for idx, c in enumerate(df.columns):
        row = idx // cols + 1  # Calculate row index for subplot
        col = idx % cols + 1   # Calculate column index for subplot

        # Create a histogram for the current column using go.Histogram
        histogram = go.Histogram(
            x=df[c], 
            nbinsx=30, 
            name=c, 
            opacity=0.75,
            marker=dict(
                color='rgb(26, 140, 255)',  # Bar color
                line=dict(
                    color='rgb(0, 0, 0)',  # Edge color (black in this case)
                    width=0.3  # Edge width
                )
            )
        )

        # Customize x-axis labels for specific columns
        if c == "Volume":
            xaxis_title = "Number of shares"
        else:
            xaxis_title = "Price in $"

        # Add histogram trace to the corresponding subplot
        fig.add_trace(histogram, row=row, col=col)

        # Update layout for the individual subplot (e.g., x-axis labels)
        fig.update_xaxes(title_text=xaxis_title, row=row, col=col)

    # Update the overall layout and appearance
    fig.update_layout(
        title="Histograms for Data Columns",
        height=900,  # Height of the overall figure
        showlegend=False,  # No legend for individual histograms
        template="plotly_dark",  # You can change the theme as needed
    )

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)

    # Boxplot of a selected column
    fig, axes = plt.subplots(1, 5, figsize=(12, 6))
    axes = axes.flatten()

    for idx, c in enumerate(df.columns):
        df[c].plot(kind='box', ax=axes[idx], color=random_color())
        axes[idx].set_title(c)

        if c == "Volume":
            axes[idx].set_yscale('log')  # Uncomment to apply log scale

    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

    # Correlation heatmap
    st.write("Correlation Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Scatter plot between two selected columns
    col1 = st.selectbox("Select the first column for scatter plot:", df.columns)
    col2 = st.selectbox("Select the second column for scatter plot:", df.columns)
    st.write(f"Scatter plot between {col1} and {col2}:")
    fig = px.scatter(df, x=col1, y=col2)
    st.plotly_chart(fig)

else:
    st.warning("Please type in a ticker to begin.")
