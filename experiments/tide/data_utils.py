from childhope.common import setup_logger
import os
import pickle
from torch.utils.data import DataLoader
from typing import Tuple, List
import torch
import numpy as np
from tqdm import tqdm
from childhope.data_processing import prepare_patient_vital_time_series

logger = setup_logger("experiments.tide.data_utils")

def prepare_data_loaders(
    source_data_dir: str,
    artifacts_dir: str,
    vital_sign_columns: List[str],
    lookback_length: int,
    horizon_length: int,
    batch_size: int,
    validation_split: float = 0.2,
    time_series_pickle_path: str = None,
    raw_data_file_name: str = "anonimized_vitals.csv"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        source_data_dir: Directory containing the data
        vital_sign_columns: List of vital signs to process
        lookback_length: Number of historical timesteps
        horizon_length: Number of future timesteps to predict
        batch_size: Batch size for DataLoaders
        validation_split: Fraction of training data to use for validation
        time_series_pickle_path: Path to pickle file for caching time series data. 
                          If None, defaults to "time_series_list.pkl" in artifacts_dir
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if time_series_pickle_path is None:
        time_series_pickle_path = os.path.join(artifacts_dir, "time_series_list.pkl")
    
    # Load or prepare time series data
    if os.path.exists(time_series_pickle_path):
        logger.info("Loading time series from pickle file...")
        with open(time_series_pickle_path, 'rb') as f:
            train_series, test_series = pickle.load(f)
        logger.info(f"Time series loaded successfully: {len(train_series)} train, {len(test_series)} test")
    else:
        logger.info("Pickle not found. Preparing time series data...")
        train_series, test_series = prepare_patient_vital_time_series(
            vitals_file_path=os.path.join(source_data_dir, raw_data_file_name),
            vital_sign_columns=vital_sign_columns,
            fill_method='linear',
            resample_frequency='60s'
        )
        
        logger.info("Saving time series to pickle file...")
        with open(time_series_pickle_path, 'wb') as f:
            pickle.dump((train_series, test_series), f)
        logger.info("Time series pickled successfully.")
    
    # Split training data into train and validation
    logger.info("Splitting training data into train and validation sets...")
    val_size = int(validation_split * len(train_series))
    val_series = train_series[-val_size:]
    train_series = train_series[:-val_size]
    logger.info(f"Split sizes: {len(train_series)} train, {len(val_series)} validation, {len(test_series)} test")
    
    # Process sequences
    logger.info("Processing training sequences...")
    train_sequences, train_targets = process_tide_sequences_in_batches(
        train_series, lookback_length, horizon_length
    )
    
    logger.info("Processing validation sequences...")
    val_sequences, val_targets = process_tide_sequences_in_batches(
        val_series, lookback_length, horizon_length
    )
    
    logger.info("Processing test sequences...")
    test_sequences, test_targets = process_tide_sequences_in_batches(
        test_series, lookback_length, horizon_length
    )

    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(train_sequences, train_targets)
    val_dataset = TimeSeriesDataset(val_sequences, val_targets)
    test_dataset = TimeSeriesDataset(test_sequences, test_targets)
    
    # Log sample data statistics
    logger.info("\nData Statistics:")
    logger.info(f"Train sequences shape: {train_sequences.shape}")
    logger.info(f"Train targets shape: {train_targets.shape}")
    
    # Log sample instances
    logger.info("\nSample Training Instance:")
    sample_seq, sample_target = train_dataset[0]
    logger.info(f"Sequence shape: {sample_seq.shape}")
    logger.info(f"Target shape: {sample_target.shape}")
    logger.info(f"Sequence stats - Min: {sample_seq.min():.3f}, Max: {sample_seq.max():.3f}, Mean: {sample_seq.mean():.3f}")
    logger.info(f"Target stats - Min: {sample_target.min():.3f}, Max: {sample_target.max():.3f}, Mean: {sample_target.mean():.3f}")
    
    # Check for NaN values
    logger.info("\nData Quality Checks:")
    logger.info(f"Train NaN count: {torch.isnan(train_sequences).sum().item()}")
    logger.info(f"Val NaN count: {torch.isnan(val_sequences).sum().item()}")
    logger.info(f"Test NaN count: {torch.isnan(test_sequences).sum().item()}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Log sample batch
    sample_batch_seq, sample_batch_target = next(iter(train_loader))
    logger.info("\nSample Batch:")
    logger.info(f"Batch sequence shape: {sample_batch_seq.shape}")
    logger.info(f"Batch target shape: {sample_batch_target.shape}")
    
    # Add detailed sequence logging
    logger.info("\nDetailed Sample Sequences:")
    for i in range(min(2, len(sample_batch_seq))):  # Show first 2 sequences
        logger.info(f"\nSequence {i+1}:")
        logger.info("Input sequence:")
        logger.info(sample_batch_seq[i].numpy())
        logger.info("\nCorresponding target:")
        logger.info(sample_batch_target[i].numpy())
    
    return train_loader, val_loader, test_loader 

def create_tide_sequences(data, lookback_length, horizon_length):
    """
    Create sequences for TiDE model training.
    
    Args:
        data: Input time series data
        lookback_length: Number of historical timesteps (L)
        horizon_length: Number of future timesteps to predict (H)
    
    Returns:
        sequences: Historical sequences [N, L, features]
        targets: Future target values [N, H, features]
    """
    sequences = []
    targets = []
    total_length = lookback_length + horizon_length
    
    for i in range(len(data) - total_length + 1):
        sequence = data[i:i + lookback_length]
        target = data[i + lookback_length:i + total_length]
        sequences.append(sequence)
        targets.append(target)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def process_tide_sequences_in_batches(time_series_list, lookback_length, horizon_length, batch_size=50):
    """
    Process sequences for TiDE in batches to avoid memory issues.
    
    Args:
        time_series_list: List of time series to process
        lookback_length: Number of historical timesteps (L)
        horizon_length: Number of future timesteps to predict (H)
        batch_size: Number of time series to process at once
    
    Returns:
        all_sequences: Concatenated sequences tensor
        all_targets: Concatenated targets tensor
    """
    all_sequences, all_targets = [], []
    
    for i in range(0, len(time_series_list), batch_size):
        batch = time_series_list[i:i + batch_size]
        batch_sequences, batch_targets = [], []
        
        for ts in tqdm(batch, desc=f"Processing batch {i//batch_size + 1}", leave=False):
            ts_values = ts.values()
            sequences, targets = create_tide_sequences(ts_values, lookback_length, horizon_length)
            batch_sequences.append(sequences)
            batch_targets.append(targets)
        
        batch_sequences = torch.cat(batch_sequences)
        batch_targets = torch.cat(batch_targets)
        
        all_sequences.append(batch_sequences)
        all_targets.append(batch_targets)
        
        del batch_sequences, batch_targets
        torch.cuda.empty_cache()
    
    return torch.cat(all_sequences), torch.cat(all_targets)

class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset class for TiDE model training.
    
    Args:
        sequences: Historical sequences [N, L, features]
        targets: Future target values [N, H, features]
    """
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]