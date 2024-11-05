import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from darts import TimeSeries

class BaseLineLSTM(nn.Module):
    """
    A baseline LSTM model for time series prediction that incorporates best practices
    while maintaining simplicity. This model serves as a strong baseline because:

    Architecture Features:
    - Layer Normalization: Helps stabilize training by normalizing inputs and hidden states
    - Xavier Initialization: A weight initialization technique that sets initial weights 
      to values that maintain equal variance throughout the network. This prevents the 
      vanishing/exploding gradient problem during training, especially in deep networks.
    - Configurable dropout: Prevents overfitting in deeper networks
    - Single linear output layer: Keeps the model simple yet effective

    Why it's a good baseline:
    1. Simplicity: Clear architecture without complex additions, making it easy to debug
       and establish a performance benchmark
    2. Robustness: Includes essential modern practices (normalization, initialization)
       without sophisticated techniques that might introduce variability
    3. Flexibility: Can handle variable sequence lengths and multiple features
    4. Memory: LSTM cells naturally capture both short and long-term dependencies in
       time series data, making them suitable for vital signs monitoring

    Technical Note:
    Xavier initialization draws weights from a distribution with its variance calculated 
    based on the number of input and output neurons. This helps ensure that the signal 
    remains in a reasonable range of values through many layers, making training more stable.

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of features in the hidden state
        num_layers (int): Number of stacked LSTM layers
        output_size (int): Number of output features
        dropout (float): Dropout rate (applied between LSTM layers if num_layers > 1)
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(BaseLineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Add Layer Normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Simple LSTM with dropout
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Add Layer Normalization after LSTM
        self.hidden_norm = nn.LayerNorm(hidden_size)
        
        # Simple linear layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Apply input normalization
        x = self.input_norm(x)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step and normalize
        out = self.hidden_norm(out[:, -1, :])
        
        # Final linear layer
        out = self.fc(out)
        
        return out