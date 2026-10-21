import torch
import torch.nn as nn

# ==========================================
# 1. Deep Neural Network (DNN)
# ==========================================
class DNNClassifier(nn.Module):
    def __init__(
        self, 
        n_features=561, 
        hidden_dim1=128, 
        hidden_dim2=64, 
        dropout_p=0.3, 
        n_classes=6, 
        **kwargs
    ):
        super(DNNClassifier, self).__init__()
        
        # Flatten just in case input comes as 3D tensor
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=n_features, out_features=hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Linear(in_features=hidden_dim2, out_features=n_classes)

    def forward(self, x):
        # ensure 2D: (batch, ...) -> (batch, 561)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        # return raw logits (batch, 6)
        out = self.classifier(x)
        return out


# ==========================================
# 2. Recurrent Neural Network (RNN)
# ==========================================
class RNNClassifier(nn.Module):
    def __init__(
        self, 
        input_size=561, 
        hidden_size=64, 
        num_layers=1, 
        dropout_p=0.0, 
        n_classes=6, 
        **kwargs
    ):
        super(RNNClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(in_features=hidden_size, out_features=n_classes)

    def forward(self, x):
        # PyTorch RNN needs 3D: (batch, seq_len=1, features=561)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        out, hidden = self.rnn(x)
        # get last hidden state h_T
        last_hidden = hidden[-1]
        logits = self.classifier(last_hidden)
        return logits


# ==========================================
# 3. Convolutional Neural Network (CNN) 1D
# ==========================================
class CNNClassifier(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        sequence_length=561, 
        num_filters=64, 
        kernel_size=3, 
        pool_size=2, 
        dropout_p=0.0, 
        n_classes=6, 
        **kwargs
    ):
        super(CNNClassifier, self).__init__()
        
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        
        self.conv1d = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=num_filters, 
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Pass a dummy tensor to figure out fc input size automatically
        # Avoids computing math by hand: (561 + 2*1 - 3)/1 + 1 = 561 -> pool(2) = 280 -> 64*280 = 17920
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, sequence_length)
            x_dummy = self.conv1d(dummy_input)
            x_dummy = self.relu(x_dummy)
            x_dummy = self.pool(x_dummy)
            x_dummy = self.flatten(x_dummy)
            flattened_dim = x_dummy.shape[1]
            
        self.classifier = nn.Linear(in_features=flattened_dim, out_features=n_classes)

    def forward(self, x):
        # reshape flat 2D input (batch, 561) to 3D Conv1D (batch, 1, 561)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# ==========================================
# 4. Long Short-Term Memory (LSTM)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(
        self, 
        input_size=561, 
        hidden_size=64, 
        num_layers=1, 
        dropout_p=0.0, 
        n_classes=6, 
        **kwargs
    ):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(in_features=hidden_size, out_features=n_classes)

    def forward(self, x):
        # add sequence dim if 2D input: (batch, 561) -> (batch, 1, 561)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        out, (h_n, c_n) = self.lstm(x)
        # take last layer hidden state
        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden)
        return logits