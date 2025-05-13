import numpy as np
import pandas as pd
import yfinance as yf

def create_sequence(data: np.ndarray, input_seq_len: int = 8, target_seq_len: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences and targets from data using the provided sequence lengths.

    Parameters
    ----------
    data : np.ndarray
        Data used to create sequences.

    input_seq_len : int, default=8
        Input sequence length.
    
    target_sequence_len : int, default=1
        Target sequence length.

    Returns
    -------
    input_sequences, output_sequences : tuple[np.ndarray, np.ndarray]
        Arrays of input and output sequences.
    """
    X = []
    y = []
    for i in range(len(data) - input_seq_len - target_seq_len):
        X.append(data[i:i+input_seq_len])
        if target_seq_len == 1:
            y.append(data[(i+input_seq_len)])
        else:
            y.append(data[(i+input_seq_len):(i+input_seq_len+target_seq_len)])

    return np.array(X), np.array(y)

def add_financial_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Adds financial features derived from the 5 basic features (Open, Close, Low, High, Volume).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with financial data.

    window : int, default=14
        Number of past data points used to calculate Simple Moving Average and Average True Range.

    Returns
    -------
    new_df : pd.DataFrame
        Dataframe with added financial features.
    """

    # Add Daily Return
    df["Daily Return"] = (df["Close"] - df["Open"]) / df["Open"]

    # Add Lagged Return
    close_pct_change = df['Close'].pct_change()
    df['Lagged Return'] = close_pct_change.shift(1).fillna(0)

    # Add Log Return
    df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log Return'] = df['Log Return'].fillna(0)

    # Add Simple Moving Average
    df[f"SMA {window}"] = df["Close"].rolling(window=window).mean().fillna(0)

    # Add Average True Range
    df["Prev_Close"] = df['Close'].shift(1).fillna(0)
    true_range = df[['High', 'Low', 'Prev_Close']].apply(lambda x: max(x["High"] - x["Low"], abs(x["High"] - x["Prev_Close"]), abs(x["Low"] - x["Prev_Close"])), axis=1)
    df[f'ATR {window}'] = true_range.rolling(window=window).mean().fillna(0)
    df.drop(["Prev_Close"], axis=1, inplace=True)

    return df

def get_financial_data(ticker: str = "AAPL", start_date: str = "2015-01-01", window: int = 14) -> pd.DataFrame:
    """
    Downloads financial data with given ticker starting at start_date.

    Parameters
    ----------
    ticker : str, default="AAPL"
        Company name.
    
    start_date : str, default = "2015-01-01"
        Starting date.

    window : int, default=14
        Lookback window used to calculate new features.

    Returns
    -------
    financial_data : pd.DataFrame
    """
    df = yf.download(ticker) # Download data
    df.columns = df.columns.droplevel(1) # Drop Ticker index
    df = add_financial_features(df, window) # Add finacial data (Returns, SMA, ATR)
    df = df[df.index >= start_date] # Select rows starting at start_date
    return df