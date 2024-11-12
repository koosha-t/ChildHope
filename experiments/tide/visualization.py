import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_predictions(model, initial_sequence, actual_values, predictions, 
                    vital_sign_columns, patient_idx, writer, device):
    """
    Plot predictions against actual values for each vital sign
    
    Args:
        model: Trained TiDE model
        initial_sequence: Input sequence used for prediction
        actual_values: Ground truth values
        predictions: Model predictions
        vital_sign_columns: List of vital sign names
        patient_idx: Patient identifier
        writer: TensorBoard writer
        device: torch device
    """
    lookback_length = initial_sequence.shape[0]
    
    for j, vital_name in enumerate(vital_sign_columns):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot initial sequence
        x_init = range(lookback_length)
        ax.plot(x_init, initial_sequence[:, j].cpu(), 
               'b-', label='Historical Values')
        
        # Plot actual values
        x_valid = range(lookback_length, lookback_length + len(actual_values))
        ax.plot(x_valid, actual_values[:, j].cpu(), 
               'g-', label='Actual Values')
        
        # Plot predictions
        ax.plot(x_valid, predictions[:, j].cpu(), 
               'r--', label='Predicted Values')
        
        ax.set_title(f'Patient {patient_idx} - {vital_name} Predictions')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        # Add to tensorboard
        writer.add_figure(
            f'Test_Predictions/Patient_{patient_idx}/{vital_name}',
            fig,
            global_step=None
        )
        plt.close(fig) 