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

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset for Pandas DataFrames
class PandasDataset(Dataset):
    def __init__(self, features_df, labels_df, num_binary_outputs=11):
        self.features = features_df.values.astype(np.float32)
        self.labels = labels_df.values.astype(np.float32)
        self.num_total_outputs = labels_df.shape[1]
        self.num_binary_outputs = num_binary_outputs
        self.num_continuous_outputs = self.num_total_outputs - self.num_binary_outputs

        if self.num_continuous_outputs < 0:
            raise ValueError("num_binary_outputs cannot be greater than total number of label columns.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        all_labels_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.num_continuous_outputs > 0 and self.num_binary_outputs > 0:
            # Both continuous and binary outputs
            continuous_labels = all_labels_tensor[:self.num_continuous_outputs]
            binary_labels = all_labels_tensor[self.num_continuous_outputs:]
            return features_tensor, (continuous_labels, binary_labels)
        elif self.num_continuous_outputs > 0 and self.num_binary_outputs == 0:
            # Only continuous outputs
            return features_tensor, all_labels_tensor
        elif self.num_continuous_outputs == 0 and self.num_binary_outputs > 0:
            # Only binary outputs (clinician case)
            return features_tensor, all_labels_tensor
        else:
            raise ValueError("Dataset configuration error: no outputs defined.")

# Load the data
X_train_df = pd.read_csv('./clinician_X_train.csv')
y_train_df = pd.read_csv('./clinician_y_train.csv')
X_test_df = pd.read_csv('./clinician_X_test.csv')
y_test_df = pd.read_csv('./clinician_y_test.csv')
X_val_df = pd.read_csv('./clinician_X_val.csv')
y_val_df = pd.read_csv('./clinician_y_val.csv')

# Drop subject_id from all data splits to prevent data leakage or irrelevant input
# Ensure 'subject_id' is dropped from feature sets (X)
if 'subject_id' in X_train_df.columns:
    X_train_df = X_train_df.drop(columns=['subject_id'])
if 'subject_id' in X_test_df.columns:
    X_test_df = X_test_df.drop(columns=['subject_id'])
if 'subject_id' in X_val_df.columns:
    X_val_df = X_val_df.drop(columns=['subject_id'])

# Ensure 'subject_id' is dropped from label sets (y) if it's not a target
if 'subject_id' in y_train_df.columns:
    y_train_df = y_train_df.drop(columns=['subject_id'])
if 'subject_id' in y_test_df.columns:
    y_test_df = y_test_df.drop(columns=['subject_id'])
if 'subject_id' in y_val_df.columns:
    y_val_df = y_val_df.drop(columns=['subject_id'])

# Align features/labels using columns_schema.json if present
schema_path = os.path.join(os.path.dirname(__file__), 'columns_schema.json')
action_cols = None
if os.path.exists(schema_path):
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        state_cols = schema.get('clinician_state_cols', [])
        # Drop absolute time from clinician inputs to match ALPACA observations
        if 'months_since_bl' in state_cols:
            state_cols = [c for c in state_cols if c != 'months_since_bl']
        action_cols = schema.get('clinician_action_cols', [])
        if state_cols:
            X_train_df = X_train_df[[c for c in state_cols if c in X_train_df.columns]]
            X_val_df = X_val_df[[c for c in state_cols if c in X_val_df.columns]]
            X_test_df = X_test_df[[c for c in state_cols if c in X_test_df.columns]]
        if action_cols:
            y_train_df = y_train_df[[c for c in action_cols if c in y_train_df.columns]]
            y_val_df = y_val_df[[c for c in action_cols if c in y_val_df.columns]]
            y_test_df = y_test_df[[c for c in action_cols if c in y_test_df.columns]]
        print("Aligned clinician data to columns_schema.json")
    except Exception as e:
        print(f"Warning: Failed to apply columns_schema.json: {e}")

# Create Dataset objects
NUM_BINARY_OUTPUTS = len(action_cols) if action_cols else y_train_df.shape[1]

train_dataset = PandasDataset(X_train_df, y_train_df, num_binary_outputs=NUM_BINARY_OUTPUTS)
test_dataset = PandasDataset(X_test_df, y_test_df, num_binary_outputs=NUM_BINARY_OUTPUTS)
val_dataset = PandasDataset(X_val_df, y_val_df, num_binary_outputs=NUM_BINARY_OUTPUTS)

print("X_train_df.columns: ", X_train_df.columns)
print("y_train_df.columns: ", y_train_df.columns)
print("X_val_df.columns: ", X_val_df.columns)
print("y_val_df.columns: ", y_val_df.columns)
# Use a dataloader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# print("Dataloader: ", train_x_loader) # Commented out or remove old loader references

print("X_train.shape:", X_train_df.shape)
print("y_train.shape:", y_train_df.shape)
print("X_test.shape:", X_test_df.shape)
print("y_test.shape:", y_test_df.shape)
print("X_val.shape:", X_val_df.shape)
print("y_val.shape:", y_val_df.shape)

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


if __name__ == "__main__":
    # Determine input and target sizes from data
    actual_input_size = X_train_df.shape[1]
    num_total_outputs = y_train_df.shape[1]
    num_binary_outputs_main = NUM_BINARY_OUTPUTS # Use the global constant
    num_continuous_outputs_main = 0  # Clinician y data contains only binary action features
    
    print(f"Input size (X features): {actual_input_size}")
    print(f"Total outputs (y features): {num_total_outputs}")
    print(f"Binary outputs (actions): {num_binary_outputs_main}")
    print(f"Continuous outputs: {num_continuous_outputs_main}")

    if num_binary_outputs_main != num_total_outputs:
        raise ValueError(f"Expected {num_binary_outputs_main} binary outputs but y_train_df has {num_total_outputs} columns.")

    model = BayesianModel(input_size=actual_input_size, 
                    num_continuous_outputs=num_continuous_outputs_main, 
                    num_binary_outputs=num_binary_outputs_main).to(device)
    
    # Since we only have binary outputs (actions), we don't need feature-specific loss weights for continuous features
    # All y outputs are binary medication decisions with equal importance
    print("No continuous outputs - using standard BCE loss for all 17 binary action features")

    num_epochs = 500

    # Define the loss function and optimizer
    criterion_bce = nn.BCEWithLogitsLoss() # For binary action outputs
    kl_weight = 0.1 # Weight for KL divergence term

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    # Early stopping parameters
    early_stop_patience = 20  # Number of epochs to wait for improvement
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model_path = './best_clinician_policy.pth'
    print(f"Early stopping patience: {early_stop_patience} epochs. Best model will be saved to {best_model_path}")

    # Train the model
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        
        for batch_idx, (features, labels_binary) in enumerate(train_loader):
            features = features.to(device)
            labels_binary = labels_binary.to(device)

            optimizer.zero_grad()
            
            # Model outputs only binary logits for actions
            _, outputs_binary_logits = model(features)
            loss_bin = criterion_bce(outputs_binary_logits, labels_binary)
            loss_kl = model.kl_loss()
            total_loss = loss_bin + kl_weight * loss_kl
            
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels_binary in val_loader:
                features = features.to(device)
                labels_binary = labels_binary.to(device)
                
                # Model outputs only binary logits for actions
                _, outputs_binary_logits = model(features)
                loss_bin = criterion_bce(outputs_binary_logits, labels_binary)
                loss_kl = model.kl_loss()
                current_batch_loss = loss_bin + kl_weight * loss_kl
                
                val_loss += current_batch_loss.item() 
        
        epoch_loss = running_loss / len(train_loader)
        current_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {current_val_loss:.4f}")

        # Step the scheduler
        scheduler.step(current_val_loss)

        # Early stopping check and save best model
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Saved model to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
            break

    # Test the model
    print(f"Loading best model from {best_model_path} for testing.")
    model.load_state_dict(torch.load(best_model_path))
    model.eval() 
    test_loss = 0.0
    with torch.no_grad(): 
        for features, labels_binary in test_loader:
            features = features.to(device)
            labels_binary = labels_binary.to(device)

            # Model outputs only binary logits for actions
            _, outputs_binary_logits = model(features)
            loss_bin = criterion_bce(outputs_binary_logits, labels_binary)
            loss_kl = model.kl_loss()
            current_batch_loss = loss_bin + kl_weight * loss_kl

            test_loss += current_batch_loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

    # Inspect input layer weights for feature importance
    print("\n--- Feature Importance Analysis (from weight_mu) ---")
    # In a Bayesian layer, we use the mean of the weight distribution (weight_mu) as a proxy for importance.
    # model.fc1.weight_mu has shape (hidden_size, input_size)
    raw_input_weights = model.fc1.weight_mu.data
    
    # Move to CPU if necessary
    if raw_input_weights.is_cuda or raw_input_weights.is_mps:
        raw_input_weights_cpu = raw_input_weights.cpu()
    else:
        raw_input_weights_cpu = raw_input_weights
    
    # Calculate importance for each input feature by summing absolute weights across hidden units.
    aggregated_abs_weights_per_feature = torch.abs(raw_input_weights_cpu).sum(dim=0)
    input_feature_importances_np = aggregated_abs_weights_per_feature.numpy()

    feature_names = X_train_df.columns
    
    if len(feature_names) != len(input_feature_importances_np):
        print(f"Warning: Number of feature names ({len(feature_names)}) does not match number of aggregated weights ({len(input_feature_importances_np)}).")
        print(f"Shapes: feature_names: {len(feature_names)}, aggregated_weights: {input_feature_importances_np.shape}")
        print(f"Original weights shape (hidden_size, input_size): {raw_input_weights.shape}")
        print("Ensure X_train_df.columns corresponds to the model's input features and aggregation logic is correct.")
    else:
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'abs_weight_sum': input_feature_importances_np})
        feature_importance_df = feature_importance_df.sort_values(by='abs_weight_sum', ascending=True)
        
        print("Features sorted by sum of absolute weights (ascending - least important first):")
        print("10 Least Important Features:")
        print(feature_importance_df.head(10))
        print("10 Most Important Features:")
        print(feature_importance_df.tail(10))
        
        threshold = 1e-3 # This threshold might need tuning for weight_mu
        potential_features_to_remove = feature_importance_df[feature_importance_df['abs_weight_sum'] < threshold]
        print(f"\nFeatures with sum of absolute weights less than {threshold} (potential candidates for removal):")
        if not potential_features_to_remove.empty:
            print(potential_features_to_remove)
        else:
            print(f"No features found with sum of absolute weights less than the current threshold ({threshold}).")
