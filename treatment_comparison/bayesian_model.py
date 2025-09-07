import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np # Added for converting pandas to numpy
import torch.nn.functional as F
import os
import json


# -------------------
# Hyperparameters & KL Annealing Settings
# -------------------
HIDDEN_DIM = 512
LEARNING_RATE = 0.001
PATIENCE = 15
KL_WEIGHT_MIN = 0.00001  # Minimum KL weight
KL_WEIGHT_MAX = 0.1    # Back to original effective weight
EPOCHS = 100
BATCH_SIZE = 32
GRAD_CLIP = 1.0
DROPOUT_RATE = 0.1
# Feature-specific loss weights (higher weight = more importance)
FEATURE_LOSS_WEIGHTS = {
    'TRABSCOR': 2.0,  # Moderate boost for TRABSCOR
    'default': 1.0    # Default weight for other features
}

# Sigmoid annealing parameters for KL weight
KL_SIGMOID_STEEPNESS = 0.6  # Controls how steeply the weight rises
mid_epoch = 50      # Center of the annealing curve

torch.manual_seed(42)

# -------------------
# Custom Bayesian Linear Layer with Updated Variational Initialization
# -------------------
class CustomBayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=2.0, scale_factor=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Make prior_sigma a learnable parameter directly
        self.prior_sigma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # NOTE: making scale factor a learnable parameter
        # self.scale_factor = scale_factor
        self.log_scale_factor = nn.Parameter(torch.tensor(np.log(scale_factor), dtype=torch.float32))
        
        # Variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Use a small variance for weight initialization
        nn.init.normal_(self.weight_mu, 0, 0.01)
        nn.init.constant_(self.bias_mu, 0.0)
        # Adjusted initialization for rho: -4 instead of -6 gives a bit more flexibility
        nn.init.constant_(self.weight_rho, -4)
        nn.init.constant_(self.bias_rho, -4)
        
    def get_sigma(self, rho):
        # Scaled softplus for stability
        return 0.05 * F.softplus(rho)
        
    def forward(self, x):
        weight_sigma = self.get_sigma(self.weight_rho)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        
        bias_sigma = self.get_sigma(self.bias_rho)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return F.linear(x, weight, bias)
        
    def kl_loss(self):
        # Use the learnable prior_sigma directly
        prior_sigma = F.softplus(self.prior_sigma) + 1e-5
        weight_sigma = self.get_sigma(self.weight_rho)
        bias_sigma = self.get_sigma(self.bias_rho)
        
        kl = 0.5 * (weight_sigma**2 / prior_sigma**2 + 
                    self.weight_mu**2 / prior_sigma**2 - 
                    1 - 2 * torch.log(weight_sigma / prior_sigma)).sum()
        
        kl += 0.5 * (bias_sigma**2 / prior_sigma**2 + 
                     self.bias_mu**2 / prior_sigma**2 - 
                     1 - 2 * torch.log(bias_sigma / prior_sigma)).sum()
        
        scale_factor = torch.exp(self.log_scale_factor)

        return kl * scale_factor

# -------------------
# Simplified Bayesian Neural Network with Targeted Improvements
# -------------------
class BayesianModel(nn.Module):
    layer_size = HIDDEN_DIM
    def __init__(self, input_size, num_continuous_outputs, num_binary_outputs, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_continuous_outputs = num_continuous_outputs
        self.num_binary_outputs = num_binary_outputs
        
        # Projection layer for the first skip connection if dimensions don't match
        if input_size != self.layer_size:
            self.input_proj = CustomBayesianLinear(input_size, self.layer_size)
        else:
            self.input_proj = nn.Identity()

        self.fc1 = CustomBayesianLinear(input_size, self.layer_size)
        self.act1 = nn.ELU()
        self.ln1 = nn.LayerNorm(self.layer_size)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = CustomBayesianLinear(self.layer_size, self.layer_size)
        self.act2 = nn.ELU()
        self.ln2 = nn.LayerNorm(self.layer_size)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # Additional layer for increased capacity
        self.fc3 = CustomBayesianLinear(self.layer_size, self.layer_size)
        self.act3 = nn.ELU()
        self.ln3 = nn.LayerNorm(self.layer_size)
        self.dropout3 = nn.Dropout(p=dropout_rate)

        # Output layers - simplified
        if self.num_continuous_outputs > 0:
            self.continuous_output = CustomBayesianLinear(self.layer_size, self.num_continuous_outputs, scale_factor=0.005)
        
        if self.num_binary_outputs > 0:
            self.binary_output = CustomBayesianLinear(self.layer_size, self.num_binary_outputs, scale_factor=0.005)
    
    def kl_loss(self):
        kl_sum = 0.0
        for module in self.modules():
            if isinstance(module, CustomBayesianLinear):
                kl_sum += module.kl_loss()
        return kl_sum

    def predict(self, x):
        """
        Performs a forward pass for inference, returning predictions.
        This method ensures the model is in evaluation mode and that no gradients are computed.
        For binary outputs, it returns probabilities by applying a sigmoid function.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self(x) # self(x) calls forward(x)

            if self.num_binary_outputs > 0:
                if self.num_continuous_outputs > 0:
                    continuous_mean, binary_logits = outputs
                    binary_probs = torch.sigmoid(binary_logits)
                    return continuous_mean, binary_probs
                else:  # Only binary outputs
                    _, binary_logits = outputs
                    binary_probs = torch.sigmoid(binary_logits)
                    # Match the forward pass return signature for consistency
                    return torch.empty(x.size(0), 0, device=x.device), binary_probs
            else:  # Only continuous outputs
                return outputs

    def forward(self, x):
        # Block 1 with residual connection
        identity = self.input_proj(x)
        out = self.act1(self.fc1(x))
        out = self.dropout1(out)
        x = self.ln1(out + identity)
        
        # Block 2 with residual connection
        identity = x
        out = self.act2(self.fc2(x))
        out = self.dropout2(out)
        x = self.ln2(out + identity)
        
        # Block 3 with residual connection
        identity = x
        out = self.act3(self.fc3(x))
        out = self.dropout3(out)
        x = self.ln3(out + identity)
        
        if self.num_binary_outputs > 0:
            binary_logits = self.binary_output(x)
            if self.num_continuous_outputs > 0:
                continuous_mean = self.continuous_output(x)
                return continuous_mean, binary_logits
            else:
                # Return empty tensor for continuous part if no continuous outputs
                return torch.empty(x.size(0), 0, device=x.device), binary_logits
        else: # only continuous outputs
            if self.num_continuous_outputs > 0:
                return self.continuous_output(x)
            else:
                # This case should ideally not be hit if input dimensions are correct
                raise ValueError("The model has no output layers defined.")
    
    def enable_mc_dropout(self):
        # Ensure dropout is active during inference
        self.train()
        
        # Disable BatchNorm updates if present
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.eval()
