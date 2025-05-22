from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
import joblib
from utils.utils import create_sequence
import numpy as np
import pandas as pd

class StockDataset(Dataset):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the data and label at the given index.
        
        Parameters:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            tuple: (data, label)
        """
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, dropout=0.1, predict_sequence = False):
        super(LSTMModel, self).__init__()

        self.predict_sequence = predict_sequence
        
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout,
                            batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        output, (h_out, c_out) = self.lstm(x)
        #torch.allclose(output[:, -1, :], h_out[-1]

        if self.predict_sequence:
            out = self.fc_out(output)
        else:
            out = self.fc_out(output[:, -1, :])
        
        return out
    
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, dropout=0.1, predict_sequence = False):
        super(GRUModel, self).__init__()

        self.predict_sequence = predict_sequence
        
        self.gru = nn.GRU(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout,
                            batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        output, h_out = self.gru(x)
        #torch.allclose(output[:, -1, :], h_out[-1]

        if self.predict_sequence:
            out = self.fc_out(output)
        else:
            out = self.fc_out(output[:, -1, :])
        
        return out
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers=6, dim_ff=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, 
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

def to_tensor_to_device(data: tuple[np.ndarray, ...], device) -> tuple[torch.Tensor, ...]:
    """
    Converts np.ndarrays into torch.Tensor objects and casts them onto a device.

    Parameters
    ----------
    data : tuple[np.ndarray]
        Tuple containing np.ndarrays.

    device : str
        Device for the torch Tensors.

    Returns
    -------
    tensors : tuple[torch.Tensor, ...]
        Tuple containing torch Tensors.
    """
    return tuple(
        (item if isinstance(item, torch.Tensor) else torch.Tensor(item)).to(device)
        for item in data
    )

def load_showcase_train_data(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Creates train data for the models.

    Parameters
    ----------
    df : pd.DataFrame
        Financial dataframe from which the train data is created.

    Returns
    -------
    train_data : tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]
        Train input and output data, test data for further processing.
    """
    X = df.to_numpy()
    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
    scaler = joblib.load('scalers/standard_scaler.save')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train, y_train = create_sequence(X_train_scaled, input_seq_len=10)
    _, y_train_multiple = create_sequence(X_train_scaled, input_seq_len=10, target_seq_len=10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = (X_train, y_train, y_train_multiple)
    X_train, y_train, y_train_multiple = to_tensor_to_device(data, device)
    return X_train, y_train, y_train_multiple, X_test_scaled

def load_custom_test_data(X_test_scaled: np.ndarray, input_seq_len: int = 10, target_seq_len: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates test data of variable input and output sequence lengths.

    Parameters
    ----------
    X_test_scaled : np.ndarray
        Scaled input data ready to be transformed into sequences.

    input_seq_len : int, default 10
        Input sequence length.

    target_seq_len : int, default 1
        Target sequence length.

    Returns
    -------
    test_data : tuple[torch.Tensor, torch.Tensor]
        Test input and output data.
    """
    X_test_seq, y_test_seq = create_sequence(X_test_scaled, input_seq_len, target_seq_len)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = (X_test_seq, y_test_seq)
    X_test_seq, y_test_seq = to_tensor_to_device(data, device)
    return X_test_seq, y_test_seq

def load_lstm_model(predict_sequence: bool = False) -> LSTMModel:
    """
    Loads a saved LSTM model.

    Parameters
    ----------
    predict_sequence : bool, default False
        Checks whether the loaded model predicts a sequence (many-to-many) or not (many-to-one).

    Returns
    -------
    lstm_model : LSTMModel
    """
    n_features = 10
    hidden_dim = 128
    num_layers = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not predict_sequence:
        dropout = 0.3
        lstm_model = LSTMModel(input_dim=n_features, 
                            output_dim=n_features, 
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            predict_sequence=predict_sequence)
        lstm_model.load_state_dict(torch.load("models/lstm_m2o_16-05-2025.pth", map_location=device))

    else:
        dropout = 0.1
        lstm_model = LSTMModel(input_dim=n_features, 
                        output_dim=n_features, 
                        hidden_dim=hidden_dim, 
                        num_layers=num_layers, 
                        dropout=dropout, 
                        predict_sequence=predict_sequence)
        lstm_model.load_state_dict(torch.load("models/lstm_m2m_16-05-2025.pth", map_location=device))

    lstm_model = lstm_model.to(device)
    return lstm_model