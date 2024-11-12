import os
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


from childhope.models import TiDE
from experiments.tide.data_utils import prepare_data_loaders

from experiments.tide.visualization import plot_predictions
from childhope.common import setup_logger

logger = setup_logger("experiments.tide.main")


def train_tide():
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Experiment configuration
    lookback_length = 60   # Number of historical timesteps used for prediction (L in TiDE paper)
    horizon_length = 15      # Number of future timesteps to predict (H in TiDE paper)
    hidden_size = 128        # Hidden dimension size for the model
    num_encoder_layers = 6   # Number of transformer encoder layers
    num_decoder_layers = 6   # Number of transformer decoder layers
    batch_size = 64          # Number of samples processed in each training iteration
    learning_rate = 0.0005    # Learning rate for optimizer
    num_epochs = 20          # Number of complete passes through the training dataset
    logger.info(
        f"experiment configuration:\n"
        f"  lookback_length: {lookback_length}\n"
        f"  horizon_length: {horizon_length}\n"
        f"  hidden_size: {hidden_size}\n"
        f"  num_encoder_layers: {num_encoder_layers}\n"
        f"  num_decoder_layers: {num_decoder_layers}\n"
        f"  batch_size: {batch_size}\n"
        f"  learning_rate: {learning_rate}\n"
        f"  num_epochs: {num_epochs}"
    )
    
    # Setup tensorboard logging
    experiment_name = "tide"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), 
                          "../tensorboard_logs",
                          experiment_name + "_" + timestamp)
    writer = SummaryWriter(log_dir)
    
    # Add hyperparameters table to tensorboard
    hyperparams = {
        'lookback_length': lookback_length,
        'horizon_length': horizon_length,
        'hidden_size': hidden_size,
        'num_encoder_layers': num_encoder_layers,
        'num_decoder_layers': num_decoder_layers,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs
    }
    # Log hyperparameters as a table
    writer.add_text('Hyperparameters', 
        '|Parameter|Value|\n|-|-|\n' + '\n'.join([f'|{k}|{v}|' for k, v in hyperparams.items()]))
    
    # Load and preprocess data
    vital_sign_columns = ['ECGHR', 'ECGRR', 'SPO2HR', 'SPO2', 'PI', 'NIBP_lower', 'NIBP_upper', 'NIBP_mean']
    
    # where to save experiment artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    # if artifacts directory does not exist, create it
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)
    
    # processed time series data
    time_series_pickle_path = os.path.join(artifacts_dir, "time_series_list.pkl")
    
    logger.info(f"Loading and preparing data loaders from {artifacts_dir}...")
    # Load and prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        source_data_dir=os.path.join(os.path.dirname(__file__), "../../data"),
        artifacts_dir=artifacts_dir,
        time_series_pickle_path=time_series_pickle_path,
        vital_sign_columns=vital_sign_columns,
        lookback_length=lookback_length,
        horizon_length=horizon_length,
        batch_size=batch_size,
        validation_split=0.2
    )

    
    # Initialize model
    input_size = len(vital_sign_columns)
    output_size = input_size
    model = TiDE(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        lookback_length=lookback_length,
        horizon_length=horizon_length,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=0.2
    ).to(device)
    
    # Setup loss function, optimizer, and learning rate scheduler
    # MSELoss (Mean Squared Error) measures the average squared difference between predictions and targets
    criterion = torch.nn.MSELoss()
    
    # Adam optimizer combines benefits of RMSprop and momentum, adapting learning rates for each parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Modified scheduler for more aggressive learning rate adjustment
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,         # factor by which the learning rate will be reduced
        patience=1,        # reduce lr if no validation loss improvement for 'patience' epochs
        verbose=True       
    )
    
    # Add gradient clipping with a larger threshold
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Increased from 1.0 to 5.0
    
    # Log training start
    logger.info("\n" + "="*80)
    logger.info(f"{'STARTING TIDE MODEL TRAINING':^80}")  # Center-aligned with width 80
    logger.info("="*80 + "\n")
    
    # Calculate and log training details
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    num_test_batches = len(test_loader)
    total_train_samples = len(train_loader.dataset)
    total_val_samples = len(val_loader.dataset)
    total_test_samples = len(test_loader.dataset)
    
    logger.info("Dataset Statistics:")
    logger.info(f"  Total samples:        {total_train_samples + total_val_samples + total_test_samples:,}")
    logger.info(f"  Training samples:     {total_train_samples:,}")
    logger.info(f"  Validation samples:   {total_val_samples:,}")
    logger.info(f"  Test samples:         {total_test_samples:,}")
    logger.info(f"  Training batches:     {num_train_batches:,}")
    logger.info(f"  Validation batches:   {num_val_batches:,}")
    logger.info(f"  Test batches:         {num_test_batches:,}")
    
    logger.info("\nTraining Configuration:")
    logger.info(f"  Total epochs:         {num_epochs}")
    logger.info(f"  Batch size:           {batch_size}")
    logger.info(f"  Learning rate:        {learning_rate}")
    logger.info(f"  Device:               {device}")
    
    logger.info("\n" + "="*80)
    logger.info(f"{'TRAINING LOOP STARTING':^80}")  # Center-aligned with width 80
    logger.info("="*80 + "\n")
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0  # Add counter for overall training steps
    previous_lr = learning_rate
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        for batch_idx, (sequences, targets) in enumerate(train_bar):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate training loss
            train_loss += loss.item()
            current_loss = train_loss / (batch_idx + 1)
            
            # Update progress bar
            train_bar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            # Optionally log batch-level training loss
            writer.add_scalar('Batch/train_loss', loss.item(), global_step)
            global_step += 1
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        with torch.no_grad():
            for val_sequences, val_targets in val_bar:
                # Move validation data to device
                val_sequences = val_sequences.to(device)
                val_targets = val_targets.to(device)
                
                val_outputs = model(val_sequences)
                batch_loss = criterion(val_outputs, val_targets).item()
                val_loss += batch_loss
                val_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
        
        val_loss /= len(val_loader)
        
        # Log epoch-level metrics with clearer names
        writer.add_scalars('Epoch Losses', {
            'train': train_loss,
            'validation': val_loss
        }, epoch)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Update learning rate and log the new value
        lr_scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Add log message for learning rate changes
        if epoch > 0 and current_lr != previous_lr:
            logger.info(f"Learning rate adjusted from {previous_lr:.6f} to {current_lr:.6f}")
        previous_lr = current_lr
        
        # Log epoch results
        logger.info(
            f'Epoch [{epoch+1}/{num_epochs}] Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}'
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(artifacts_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved! Validation loss improved to {val_loss:.6f}")
            logger.info(f"Model saved to: {model_path}")
        
    writer.close()
    
    # Load best model for test evaluation
    logger.info("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, 'best_model.pth')))
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Final Test Loss: {avg_test_loss:.6f}")
    
    # Visualization of predictions
    logger.info("Generating predictions for visualization...")
    
    def predict_sequence(model, initial_sequence, num_steps):
        """Generate multi-step predictions"""
        model.eval()
        current_sequence = initial_sequence.clone()
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_steps):
                output = model(current_sequence.unsqueeze(0))
                pred = output[:, -1, :]  # Take last prediction
                predictions.append(pred)
                
                # Update sequence for next prediction
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    pred.unsqueeze(1)
                ], dim=1)
        
        return torch.cat(predictions, dim=0)
    
    # Select random test batches for visualization
    num_samples = 5
    test_samples = []
    
    # Collect a few test samples
    with torch.no_grad():
        for sequences, targets in test_loader:
            test_samples.append((sequences[0], targets[0]))  # Take first sequence from each batch
            if len(test_samples) >= num_samples:
                break
    
    for idx, (sequence, target) in enumerate(test_samples):
        # Move data to device
        sequence = sequence.to(device)
        
        # Generate predictions
        predictions = predict_sequence(
            model,
            sequence.unsqueeze(0),
            num_steps=horizon_length*5  # Predict 5 horizons
        ).cpu().numpy()
        
        # Get historical data and ground truth
        historical_data = sequence.cpu().numpy()
        ground_truth = target.cpu().numpy()
        
        # Create visualization for each vital sign
        for j, vital_name in enumerate(vital_sign_columns):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(range(-len(historical_data), 0), 
                   historical_data[:, j], 
                   label='Historical', 
                   color='blue')
            
            # Plot ground truth
            ax.plot(range(len(ground_truth)), 
                   ground_truth[:, j], 
                   label='Ground Truth', 
                   color='green')
            
            # Plot predictions
            ax.plot(range(len(predictions)), 
                   predictions[:, j], 
                   label='Predictions', 
                   color='red', 
                   linestyle='--')
            
            ax.set_title(f'Sample {idx} - {vital_name} Predictions')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(log_dir, f'sample_{idx}_{vital_name}_predictions.png'))
            plt.close()

        logger.info(f"Saved prediction plots for sample {idx}")

if __name__ == "__main__":
    train_tide() 