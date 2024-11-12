import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from childhope.models import TiDE
from experiments.tide.data_utils import prepare_data_loaders
from childhope.common import setup_logger

logger = setup_logger("experiments.tide.visualize")

def visualize_predictions():
    # Configuration - make sure these match your training configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vital_sign_columns = ['ECGHR', 'ECGRR', 'SPO2HR', 'SPO2', 'PI', 'NIBP_lower', 'NIBP_upper', 'NIBP_mean']
    lookback_length = 60
    horizon_length = 15
    hidden_size = 128
    num_encoder_layers = 6
    num_decoder_layers = 6
    batch_size = 64
    num_iterations = 4

    # Paths
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    time_series_pickle_path = os.path.join(artifacts_dir, "time_series_list.pkl")
    model_path = os.path.join(artifacts_dir, 'best_model.pth')
    vis_output_dir = os.path.join(artifacts_dir, 'visualizations')
    os.makedirs(vis_output_dir, exist_ok=True)

    # Load data
    _, _, test_loader = prepare_data_loaders(
        source_data_dir=os.path.join(os.path.dirname(__file__), "../../data"),
        artifacts_dir=artifacts_dir,
        time_series_pickle_path=time_series_pickle_path,
        vital_sign_columns=vital_sign_columns,
        lookback_length=lookback_length,
        horizon_length=horizon_length,
        batch_size=batch_size,
        validation_split=0.2
    )

    # Initialize and load model
    input_size = len(vital_sign_columns)
    model = TiDE(
        input_size=input_size,
        output_size=input_size,
        hidden_size=hidden_size,
        lookback_length=lookback_length,
        horizon_length=horizon_length,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=0.2
    ).to(device)

    logger.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Select test samples and generate predictions
    num_samples = 20
    test_samples = []
    
    logger.info("Collecting test samples...")
    with torch.no_grad():
        for sequences, targets in test_loader:
            # Get 1-hour ground truth by taking 4 consecutive samples
            full_sequence = sequences[0]
            full_target = []
            
            # Collect 4 consecutive samples to get 1 hour of ground truth
            for i in range(4):
                if i + 1 >= len(sequences):
                    break
                full_target.append(targets[i])
            
            # Only use samples where we have full 1-hour ground truth
            if len(full_target) == 4:
                full_target = torch.cat(full_target, dim=0)
                test_samples.append((full_sequence, full_target))
                if len(test_samples) >= num_samples:
                    break

    logger.info("Generating predictions and visualizations...")
    for idx, (sequence, target) in enumerate(test_samples):
        sequence = sequence.to(device)
        target = target.to(device)
        
        # Initialize arrays for storing longer predictions
        extended_predictions = np.zeros((horizon_length * num_iterations, len(vital_sign_columns)))
        extended_ground_truth = target.cpu().numpy()  # Now contains full 1-hour ground truth
        current_sequence = sequence.clone()
        
        # Generate iterative predictions
        for iter_idx in range(num_iterations):
            with torch.no_grad():
                step_predictions = model(current_sequence.unsqueeze(0)).squeeze(0).cpu().numpy()
                
                # Store predictions
                start_idx = iter_idx * horizon_length
                end_idx = (iter_idx + 1) * horizon_length
                extended_predictions[start_idx:end_idx] = step_predictions
                
                # Update sequence for next iteration
                current_sequence = torch.cat([
                    current_sequence[horizon_length:],
                    torch.tensor(step_predictions, device=device)
                ], dim=0)
        
        # Get historical data
        historical_data = sequence.cpu().numpy()

        # Create visualization for each vital sign
        for j, vital_name in enumerate(vital_sign_columns):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(range(-len(historical_data), 0), 
                   historical_data[:, j], 
                   label='Historical', 
                   color='blue')
            
            # Plot extended predictions
            ax.plot(range(len(extended_predictions)), 
                   extended_predictions[:, j], 
                   label='Predictions', 
                   color='red', 
                   linestyle='--')
            
            # Plot full hour of ground truth
            ax.plot(range(len(extended_ground_truth)), 
                   extended_ground_truth[:, j], 
                   label='Ground Truth', 
                   color='green')
            
            ax.set_title(f'Sample {idx} - {vital_name} Predictions (1 hour)')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(vis_output_dir, f'sample_{idx}_{vital_name}_predictions.png'))
            plt.close()

        logger.info(f"Saved prediction plots for sample {idx}")

if __name__ == "__main__":
    visualize_predictions() 