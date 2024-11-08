import torch
import torch.nn as nn

################################################
#                                              #
#     TiDE: Time-series Dense Encoder          #
#     Paper: https://arxiv.org/abs/2304.08424  #
#                                              #
################################################
"""
TiDE: Time-series Dense Encoder
------------------------------
TiDE is a neural network architecture designed for long-term time-series forecasting. 
It aims to capture temporal patterns in sequential data by leveraging dense encoding 
and decoding layers, along with residual connections and temporal decoding mechanisms.

Architecture Components
----------------------
1. Feature Projection: 
   Projects input features into a higher-dimensional space to capture complex relationships.

2. Dense Encoder: 
   Encodes the input features into a latent representation.

3. Dense Decoder: 
   Decodes the latent representation back into a format suitable for forecasting.

4. Temporal Decoder: 
   Processes each time step individually to produce the final predictions.

5. Residual Connections: 
   Enhance learning by allowing gradients to flow through skip connections, 
   mitigating the vanishing gradient problem.

Key Concepts & Implementation Details
-----------------------------------

1. Feature Projection
   Why Project Features?
   - To transform input data into a higher-dimensional space where complex patterns may be more easily captured.
   - Helps in capturing non-linear relationships between features.

2. Residual Connections
   Purpose:
   - Facilitate the flow of gradients during backpropagation, reducing the vanishing gradient problem.
   - Allow the model to preserve input information by learning to pass signals unchanged when needed.

   Implementation in ResidualBlock:
   - The input (x) is added to the output after the two linear layers and dropout.
   - If the input and output dimensions differ, a linear layer adjusts the input before adding.

3. Layer Normalization
   Purpose:
   - Stabilizes the learning process by normalizing the outputs of each layer.
   - Helps the model converge faster and improves generalization.

4. Temporal Decoding
   Why Process Each Time Step Individually?
   - Time-series forecasting often requires predictions at each future time step.
   - By processing each time step separately, the model can focus on temporal dependencies specific to that time step.

5. Static Attributes
   Purpose:
   - Incorporate additional information that does not change over time but may influence the forecast (e.g., product category, location).
   - Enhances the model's ability to make accurate predictions by considering all relevant factors.
"""



class ResidualBlock(nn.Module):
    """
    Purpose: Serves as the basic building block of the TiDE architecture, implementing 
    a residual connection around a two-layer feedforward network.

    Structure:
    - Linear Layer 1: Projects the input to a higher-dimensional space (hidden_dim).
    - ReLU Activation: Introduces non-linearity.
    - Linear Layer 2: Projects back to the output dimension (output_dim).
    - Dropout: Prevents overfitting by randomly dropping units during training.
    - Residual Connection: Adds the input (x) to the output of the dropout layer.
    - Layer Normalization: Normalizes the output for stable and efficient training.
    
    Key Points:
    - Residual Connection: Adjusts for dimension mismatch between input and output.
    - Layer Normalization: Applied after adding the residual to stabilize training.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Adjust residual connection if input and output dimensions differ
        self.residual_connection = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = out + residual
        out = self.layer_norm(out)
        return out

class DenseEncoder(nn.Module):
    """
    **Dense encoder component**
    Processes concatenated features through multiple residual blocks
    to capture temporal patterns and encode the input into a latent representation.
    
    Structure:
    - Comprises multiple (num_layers) ResidualBlocks.
    - Each block takes the output of the previous block as input.
    
    Key Points:
    - Expansion Factor: Controls the size of the hidden layer in the residual block (hidden_dim * expansion_factor).
    - Same Input and Output Dimensions: Maintains consistent dimensions throughout the encoder layers.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                ResidualBlock(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    hidden_dim=hidden_dim * expansion_factor,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DenseDecoder(nn.Module):
    """
    **Dense decoder component**
    Maps encoded representations back to temporal space,
    preparing for per-timestep predictions.
    
    Structure:
    - Similar to DenseEncoder but with the possibility of changing the output dimension in the last layer.
    
    Key Points:
    - Output Dimension Adjustment: The last layer adjusts the output dimension to match the expected size.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ResidualBlock(
                    input_dim=input_dim,
                    output_dim=output_dim if i == num_layers - 1 else input_dim,
                    hidden_dim=hidden_dim * expansion_factor,
                    dropout=dropout,
                )
            )
            input_dim = output_dim if i == num_layers - 1 else input_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TemporalDecoder(nn.Module):
    """
    **Temporal decoder component**
    Processes each time step individually to convert the decoded representations into final predictions.
    
    Structure:
    - Contains one or more residual blocks that operate per time step.
    
    Key Points:
    - Per-Time-Step Processing: Operates along the time dimension to produce predictions for each future time step.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                ResidualBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim * expansion_factor,
                    dropout=dropout,
                )
            )
            input_dim = output_dim  # Update input_dim if output_dim changes

    def forward(self, x):
        # x shape: [batch, horizon, input_dim]
        for layer in self.layers:
            x = layer(x)
        return x

class TiDE(nn.Module):
    """
    TiDE (Time-series Dense Encoder) implementation.
    
    Architecture Flow (bottom to top):
    1. Dynamic Covariates → Feature Projection (per time-step)
    2. Concat [Lookback, Attributes, Projected Covariates] → Flatten
    3. Dense Encoder (×nc residual blocks)
    4. Dense Decoder (×nd residual blocks)
    5. Unflatten & Stack with projected covariates
    6. Temporal Decoder (per time-step)
    7. Add Residual connection from Lookback
    8. Final Predictions

    Args:
        input_size: Number of input features (dimension of dynamic covariates)
        output_size: Number of output features to predict
        hidden_size: Hidden dimension for feature projections and dense layers
        lookback_length: Number of historical timesteps (L)
        horizon_length: Number of future timesteps to predict (H)
        num_encoder_layers: Number of residual blocks in encoder (nc)
        num_decoder_layers: Number of residual blocks in decoder (nd)
        attribute_size: Dimension of static attributes (a)
        expansion_factor: Hidden layer expansion factor in residual blocks
        dropout: Dropout probability
        has_future_covariates: Whether to use future covariates
    """
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        lookback_length,
        horizon_length,
        num_encoder_layers=3,
        num_decoder_layers=3,
        attribute_size=0,
        expansion_factor=4,
        dropout=0.1,
        has_future_covariates=False,
    ):
        super().__init__()
        self.lookback_length = lookback_length
        self.horizon_length = horizon_length
        self.hidden_size = hidden_size
        self.has_attributes = attribute_size > 0
        self.has_future_covariates = has_future_covariates

        # Feature projection
        self.feature_projection = nn.Linear(input_size, hidden_size)

        # Optional attribute projection
        if self.has_attributes:
            self.attribute_projection = ResidualBlock(
                input_dim=attribute_size,
                output_dim=hidden_size,
                hidden_dim=hidden_size * expansion_factor,
                dropout=dropout,
            )

        # Dense Encoder
        self.encoder = DenseEncoder(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            num_layers=num_encoder_layers,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

        # Dense Decoder
        self.decoder = DenseDecoder(
            input_dim=hidden_size,
            output_dim=hidden_size,
            hidden_dim=hidden_size,
            num_layers=num_decoder_layers,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

        # Temporal Decoder
        temporal_decoder_input_dim = hidden_size
        if self.has_future_covariates:
            temporal_decoder_input_dim += hidden_size  # Assuming projected future covariates

        self.temporal_decoder = TemporalDecoder(
            input_dim=temporal_decoder_input_dim,
            output_dim=output_size,
            hidden_dim=hidden_size,
            num_layers=1,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

        # Lookback skip connection
        self.lookback_skip = nn.Linear(lookback_length * input_size, horizon_length * output_size)

    def forward(self, lookback, future_covariates=None, attributes=None):
        """
        Forward pass following TiDE architecture diagram.

        Args:
            lookback: Historical values [batch, L, input_size] (y_{1:L})
            future_covariates: Future covariates [batch, H, input_size] (x_{L+1:L+H})
            attributes: Static attributes [batch, attribute_size] (a)

        Returns:
            predictions: Forecasted values [batch, H, output_size] (ŷ_{L+1:L+H})
        
        Architecture Steps:
        1. Project lookback and future features through feature projection
        2. Concatenate lookback, attributes, and projected covariates
        3. Process through dense encoder blocks
        4. Process through dense decoder blocks
        5. Unflatten and stack with projected covariates
        6. Apply temporal decoder per time-step
        7. Add residual connection from lookback
        8. Return final predictions
        """
        batch_size = lookback.size(0)

        # Project lookback features
        lookback_projected = self.feature_projection(lookback)  # [batch, lookback_length, hidden_size]

        # Take the last time step from lookback
        last_lookback = lookback_projected[:, -1, :]  # [batch, hidden_size]

        # Project attributes if available
        if self.has_attributes and attributes is not None:
            attributes_projected = self.attribute_projection(attributes)  # [batch, hidden_size]
            encoder_input = last_lookback + attributes_projected  # [batch, hidden_size]
        else:
            encoder_input = last_lookback  # [batch, hidden_size]

        # Encode
        encoder_output = self.encoder(encoder_input)  # [batch, hidden_size]

        # Decode
        decoder_output = self.decoder(encoder_output)  # [batch, hidden_size]

        # Prepare temporal decoder input
        decoder_output_expanded = decoder_output.unsqueeze(1).expand(-1, self.horizon_length, -1)  # [batch, horizon_length, hidden_size]

        if self.has_future_covariates and future_covariates is not None:
            # Project future covariates
            future_covariates_projected = self.feature_projection(future_covariates)  # [batch, horizon_length, hidden_size]
            temporal_decoder_input = torch.cat([decoder_output_expanded, future_covariates_projected], dim=-1)  # [batch, horizon_length, combined_size]
        else:
            temporal_decoder_input = decoder_output_expanded  # [batch, horizon_length, hidden_size]

        # Temporal decoding
        predictions = self.temporal_decoder(temporal_decoder_input)  # [batch, horizon_length, output_size]

        # Add lookback residual connection
        lookback_flat = lookback.reshape(batch_size, -1)  # [batch, lookback_length * input_size]
        lookback_residual = self.lookback_skip(lookback_flat)  # [batch, horizon_length * output_size]
        lookback_residual = lookback_residual.view(batch_size, self.horizon_length, -1)  # [batch, horizon_length, output_size]

        # Sum predictions and residual
        predictions = predictions + lookback_residual  # [batch, horizon_length, output_size]

        return predictions
