from childhope.models import BaseLineLSTM
from childhope.data_processing import prepare_patient_vital_time_series
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import os
import numpy as np
from tqdm import tqdm
import pickle
from childhope.common import setup_logger

logger = setup_logger("experiments.baseline_lstm")

# Function to create sequences from time series data
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(sequence)
        targets.append(target)
    
    # Convert lists to numpy arrays before converting to tensors
    sequences = np.array(sequences)
    targets = np.array(targets)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# Add this function at the top of baseline.py
def normalize_sequences(sequences):
    """Normalize sequences using mean and std across all dimensions"""
    # Reshape to (total_sequences * sequence_length, features)
    flat_sequences = sequences.reshape(-1, sequences.shape[-1])
    
    # Calculate mean and std
    mean = torch.mean(flat_sequences, dim=0)
    std = torch.std(flat_sequences, dim=0)
    
    # Add small epsilon to avoid division by zero
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    
    # Normalize
    normalized = (sequences - mean) / std
    
    return normalized, mean, std

def process_sequences_in_batches(time_series_list, sequence_length, batch_size=100):
    """Process sequences in batches to avoid memory issues"""
    all_sequences, all_targets = [], []
    
    for i in range(0, len(time_series_list), batch_size):
        batch = time_series_list[i:i + batch_size]
        batch_sequences, batch_targets = [], []
        
        for ts in tqdm(batch, desc=f"Processing batch {i//batch_size + 1}", leave=False):
            ts_values = ts.values()
            sequences, targets = create_sequences(ts_values, sequence_length)
            batch_sequences.append(sequences)
            batch_targets.append(targets)
        
        # Stack batch
        batch_sequences = torch.cat(batch_sequences)
        batch_targets = torch.cat(batch_targets)
        
        all_sequences.append(batch_sequences)
        all_targets.append(batch_targets)
        
        # Clear some memory
        del batch_sequences, batch_targets
        torch.cuda.empty_cache()
    
    return torch.cat(all_sequences), torch.cat(all_targets)

if __name__=="__main__":
    
    # is GPU available?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    vital_sign_columns = ['ECGHR', 'ECGRR', 'SPO2HR', 'SPO2', 'PI', 'NIBP_lower', 'NIBP_upper', 'NIBP_mean']
    
    logger.info("Preparing patient vital time series data")
    
    # Define paths for pickled data
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    time_series_pickle = os.path.join(data_dir, "time_series_list.pkl")
    
    # Check if pickled time series exists
    if os.path.exists(time_series_pickle):
        logger.info("Loading time series from pickle file...")
        with open(time_series_pickle, 'rb') as f:
            time_series_list = pickle.load(f)
        logger.info("Time series loaded successfully from pickle.")
    else:
        logger.info("Pickle not found. Preparing time series data...")
        time_series_list = prepare_patient_vital_time_series(
            vitals_file_path= os.path.join(os.path.dirname(__file__), "../data/anonimized_vitals.csv"),
            vital_sign_columns=vital_sign_columns,
            fill_method='linear',
            resample_frequency='60s'
        )
        
        # Save to pickle
        logger.info("Saving time series to pickle file...")
        with open(time_series_pickle, 'wb') as f:
            pickle.dump(time_series_list, f)
        logger.info("Time series pickled successfully.")
            
    # Process sequences in batches
    logger.info("Creating sequences and targets...")
    sequence_length = 120
    all_sequences, all_targets = process_sequences_in_batches(
        time_series_list, 
        sequence_length,
        batch_size=50  # Adjust this based on available memory
    )
    logger.info("Sequences and targets created.")
    
    batch_size = 128
    
    # Create DataLoader with raw (non-normalized) data
    train_dataset = TensorDataset(all_sequences, all_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define model, loss, and optimizer
    input_size = len(vital_sign_columns)
    hidden_size = 64
    num_layers = 2
    output_size = input_size
    model = BaseLineLSTM(input_size, hidden_size, num_layers, output_size, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    try:
        model_summary = summary(model, input_size=(batch_size, sequence_length, input_size))
        logger.info(f"Model Summary:\n{model_summary}")
    except Exception as e:
        logger.warning("Unable to generate model summary with torchinfo.")
    
    num_epochs = 10
    model.train()
    logger.info("Training is about to start. Epochs: %s", num_epochs)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            # Check for NaN in inputs
            if torch.isnan(inputs).any():
                logger.error("NaN detected in inputs")
                continue
                
            optimizer.zero_grad()
            
            try:
                outputs = model(inputs)
                
                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    logger.error("NaN detected in model outputs")
                    continue
                    
                loss = criterion(outputs, targets)
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    logger.error("NaN detected in loss calculation")
                    continue
                    
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            except RuntimeError as e:
                logger.error(f"Runtime error during training: {str(e)}")
                continue
        
        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")