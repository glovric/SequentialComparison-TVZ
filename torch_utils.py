from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import joblib
from utils import create_sequence

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

def to_tensor_to_device(data, device):
    return tuple(
        (item if isinstance(item, torch.Tensor) else torch.Tensor(item)).to(device)
        for item in data
    )

def load_showcase_train_data(df):
    X = df.to_numpy()
    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
    scaler = joblib.load('scalers/standard_scaler.save')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_seq, y_train_seq = create_sequence(X_train_scaled, input_seq_len=10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = (X_train_seq, y_train_seq)
    X_train_seq, y_train_seq = to_tensor_to_device(data, device)
    return X_train_seq, y_train_seq, X_test_scaled

def load_custom_test_data(X_test_scaled, seq_len=10):
    X_test_seq, y_test_seq = create_sequence(X_test_scaled, input_seq_len=seq_len)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = (X_test_seq, y_test_seq)
    X_test_seq, y_test_seq = to_tensor_to_device(data, device)
    return X_test_seq, y_test_seq

def load_lstm_model(predict_sequence=False):
    n_features = 10
    hidden_dim = 128
    num_layers = 3

    if not predict_sequence:
        dropout = 0.3
        lstm_model = LSTMModel(input_dim=n_features, 
                            output_dim=n_features, 
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            predict_sequence=predict_sequence)
        lstm_model.load_state_dict(torch.load("models/lstm_m2o_16-05-2025.pth"))

    else:
        dropout = 0.1
        lstm_model = LSTMModel(input_dim=n_features, 
                        output_dim=n_features, 
                        hidden_dim=hidden_dim, 
                        num_layers=num_layers, 
                        dropout=dropout, 
                        predict_sequence=predict_sequence)
        lstm_model.load_state_dict(torch.load("models/lstm_m2m_16-05-2025.pth"))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lstm_model = lstm_model.to(device)
    return lstm_model

    
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