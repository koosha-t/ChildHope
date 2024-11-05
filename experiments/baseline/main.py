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
from torch.utils.tensorboard import SummaryWriter
import datetime

logger = setup_logger("experiments.baseline_lstm")

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(sequence)
        targets.append(target)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def normalize_data(sequences, targets=None, params=None):
    """
    Normalize sequences and optionally targets using mean and std.
    If params is provided, use them instead of calculating new ones.
    """
    if params is None:
        flat_sequences = sequences.reshape(-1, sequences.shape[-1])
        mean = flat_sequences.mean(dim=0)
        std = flat_sequences.std(dim=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        params = {'mean': mean, 'std': std}
    
    normalized_sequences = (sequences - params['mean']) / params['std']
    
    normalized_targets = None
    if targets is not None:
        normalized_targets = (targets - params['mean']) / params['std']
    
    return normalized_sequences, normalized_targets, params

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
        
        batch_sequences = torch.cat(batch_sequences)
        batch_targets = torch.cat(batch_targets)
        
        all_sequences.append(batch_sequences)
        all_targets.append(batch_targets)
        
        del batch_sequences, batch_targets
        torch.cuda.empty_cache()
    
    return torch.cat(all_sequences), torch.cat(all_targets)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    vital_sign_columns = ['ECGHR', 'ECGRR', 'SPO2HR', 'SPO2', 'PI', 'NIBP_lower', 'NIBP_upper', 'NIBP_mean']
    
    data_dir = os.path.join(os.path.dirname(__file__), "experiment_data")
    time_series_pickle = os.path.join(data_dir, "time_series_list.pkl")
    
    if os.path.exists(time_series_pickle):
        logger.info("Loading time series from pickle file...")
        with open(time_series_pickle, 'rb') as f:
            train_series, test_series = pickle.load(f)
        logger.info(f"Time series loaded successfully: {len(train_series)} train, {len(test_series)} test")
    else:
        logger.info("Pickle not found. Preparing time series data...")
        train_series, test_series = prepare_patient_vital_time_series(
            vitals_file_path=os.path.join(os.path.dirname(__file__), "../../data/anonimized_vitals.csv"),
            vital_sign_columns=vital_sign_columns,
            fill_method='linear',
            resample_frequency='60s'
        )
        
        logger.info("Saving time series to pickle file...")
        with open(time_series_pickle, 'wb') as f:
            pickle.dump((train_series, test_series), f)
        logger.info("Time series pickled successfully.")
    
    # Split training data into train and validation
    logger.info("Splitting training data into train and validation sets...")
    val_size = int(0.2 * len(train_series))
    val_series = train_series[-val_size:]
    train_series = train_series[:-val_size]
    logger.info(f"Split sizes: {len(train_series)} train, {len(val_series)} validation, {len(test_series)} test")

    # Process sequences for all sets
    sequence_length = 120
    logger.info("Creating sequences and targets...")
    
    train_sequences, train_targets = process_sequences_in_batches(
        train_series, sequence_length, batch_size=50
    )
    logger.info(f"Training sequences created: {train_sequences.shape}")
    
    val_sequences, val_targets = process_sequences_in_batches(
        val_series, sequence_length, batch_size=50
    )
    logger.info(f"Validation sequences created: {val_sequences.shape}")
    
    test_sequences, test_targets = process_sequences_in_batches(
        test_series, sequence_length, batch_size=50
    )
    logger.info(f"Test sequences created: {test_sequences.shape}")

    # Normalize all sets using training statistics
    logger.info("Normalizing sequences...")
    train_sequences_norm, train_targets_norm, norm_params = normalize_data(train_sequences, train_targets)
    val_sequences_norm, val_targets_norm, _ = normalize_data(val_sequences, val_targets, params=norm_params)
    test_sequences_norm, test_targets_norm, _ = normalize_data(test_sequences, test_targets, params=norm_params)
    
    torch.save(norm_params, os.path.join(data_dir, "normalization_params.pt"))
    logger.info("Normalization parameters saved.")

    # Create DataLoaders
    batch_size = 128
    train_dataset = TensorDataset(train_sequences_norm, train_targets_norm)
    val_dataset = TensorDataset(val_sequences_norm, val_targets_norm)
    test_dataset = TensorDataset(test_sequences_norm, test_targets_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model setup
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
    
    # Training parameters
    num_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    logger.info("Training is about to start. Epochs: %s", num_epochs)
    
    # Create a more structured log directory
    experiment_name = "baseline_lstm_2_layers"  # Change this for different experiments
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(os.path.dirname(__file__), "../tensorboard_logs", experiment_name + "_" + timestamp)
    writer = SummaryWriter(tb_log_dir)
    logger.info(f"TensorBoard logs will be saved to {tb_log_dir}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if torch.isnan(inputs).any():
                logger.error("NaN detected in inputs")
                continue
                
            optimizer.zero_grad()
            
            try:
                outputs = model(inputs)
                
                if torch.isnan(outputs).any():
                    logger.error("NaN detected in model outputs")
                    continue
                    
                loss = criterion(outputs, targets)
                
                if torch.isnan(loss):
                    logger.error("NaN detected in loss calculation")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Log batch-level training loss
                writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
                
                train_loss += loss.item() * inputs.size(0)
                
            except RuntimeError as e:
                logger.error(f"Runtime error during training: {str(e)}")
                continue
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                try:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                except RuntimeError as e:
                    logger.error(f"Runtime error during validation: {str(e)}")
                    continue
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Log epoch-level metrics with clearer names
        writer.add_scalars('Training Progress', {
            'Training Loss': train_loss,
            'Validation Loss': val_loss
        }, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(data_dir, "best_model.pt"))
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break
    
    # Final evaluation on test set
    logger.info("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(data_dir, "best_model.pt")))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            try:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
            except RuntimeError as e:
                logger.error(f"Runtime error during testing: {str(e)}")
                continue
    
    test_loss = test_loss / len(test_loader.dataset)
    logger.info(f"Final Test Loss: {test_loss:.4f}")
    logger.info("Training completed.")

    # Log example predictions periodically (optional)
    if epoch % 5 == 0:  # Every 5 epochs
        writer.add_scalars('Example_Predictions', {
            'true': targets[0, 0].item(),
            'predicted': outputs[0, 0].item()
        }, epoch)

    # Log final test loss
    writer.add_scalar('Loss/test', test_loss, epoch)

    # Close the writer
    writer.close()
    logger.info("TensorBoard logging completed")