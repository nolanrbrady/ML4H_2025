import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

# Adjust path to import BayesianModel from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bayesian_model import BayesianModel

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'best_bayesian_model.pth')
X_VAL_PATH = os.path.join(SCRIPT_DIR, '..', 'X_val.csv')
Y_VAL_PATH = os.path.join(SCRIPT_DIR, '..', 'y_val.csv')
NUM_MC_SAMPLES = 1000  # Number of Monte Carlo samples for uncertainty estimation
NUM_BINARY_OUTPUTS = 11 # As defined in the model training script

# --- Load Data ---
X_val_df = pd.read_csv(X_VAL_PATH)
y_val_df = pd.read_csv(Y_VAL_PATH)

# Drop subject_id if it exists
if 'subject_id' in X_val_df.columns:
    X_val_df = X_val_df.drop(columns=['subject_id'])
if 'subject_id' in y_val_df.columns:
    y_val_df = y_val_df.drop(columns=['subject_id'])

# --- Model and Data Preparation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Determine model parameters from data
input_size = X_val_df.shape[1]
num_total_outputs = y_val_df.shape[1]
num_continuous_outputs = num_total_outputs - NUM_BINARY_OUTPUTS

# Instantiate the model
model = BayesianModel(
    input_size=input_size,
    num_continuous_outputs=num_continuous_outputs,
    num_binary_outputs=NUM_BINARY_OUTPUTS
)

# Load the trained model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# Prepare validation data tensor
X_val_tensor = torch.tensor(X_val_df.values, dtype=torch.float32).to(device)

# --- SHAP Analysis ---
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Starting SHAP analysis...")

# Define a wrapper for the model to concatenate the outputs
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        continuous_mean, binary_logits = self.model(x)
        return torch.cat([continuous_mean, binary_logits], dim=1)

# Wrap the model
wrapped_model = ModelWrapper(model)
wrapped_model.to(device)
wrapped_model.eval()

# Using KernelExplainer due to custom layers in the model.
# KernelExplainer is model-agnostic but slower than DeepExplainer.
print("Using KernelExplainer for SHAP analysis. This may be slow.")

# 1. Create a prediction function for the explainer
def predict_fn(x_numpy):
    # shap can pass 1D array for a single sample, so we ensure it's 2D
    if x_numpy.ndim == 1:
        x_numpy = x_numpy.reshape(1, -1)
    
    # Ensure model is in eval mode for consistent predictions if it has dropout etc.
    wrapped_model.eval()
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = wrapped_model(x_tensor)
    return predictions.cpu().numpy()

# Manually check the prediction function's output shape to get the correct number of outputs
test_pred = predict_fn(X_val_df.head(1).values)
num_total_outputs = test_pred.shape[1]
print(f"Detected {num_total_outputs} outputs from the model.")

# 2. Summarize the background data. A smaller summary (e.g., 50 k-means centroids) is recommended.
# Using the first 100 validation samples as background for summarization
background_data_for_summary = X_val_df.head(100).values
background_summary = shap.kmeans(background_data_for_summary, 50)

# 3. Create the KernelExplainer
explainer = shap.KernelExplainer(predict_fn, background_summary)

# 4. Explain a small number of samples due to performance.
X_explain_df = X_val_df.head(5)
print(f"Calculating SHAP values for {len(X_explain_df)} samples...")

# KernelExplainer's sampling process inherently handles model stochasticity.
# The result is a list of arrays, one for each sample, with shape (n_features, n_outputs)
shap_values_by_sample = explainer.shap_values(X_explain_df.values)

# Restructure SHAP values to be grouped by output, not by sample
n_samples = len(X_explain_df)
n_features = X_explain_df.shape[1]
shap_values_by_output = [np.zeros((n_samples, n_features)) for _ in range(num_total_outputs)]
for i in range(num_total_outputs):
    for j in range(n_samples):
        shap_values_by_output[i][j, :] = shap_values_by_sample[j][:, i]

# 5. Get feature and output names
feature_names = X_val_df.columns.tolist()
output_names = y_val_df.columns.tolist()

# 6. Visualize the SHAP values
# Create a directory to save SHAP plots
SHAP_PLOTS_DIR = os.path.join(SCRIPT_DIR, 'shap_plots')
os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)
print(f"Saving SHAP plots to: {SHAP_PLOTS_DIR}")

# Create summary plots for each output
for i in range(num_total_outputs):
    # Ensure output name exists, otherwise use a generic name
    output_name = output_names[i] if i < len(output_names) else f"Output_{i+1}"
    # Sanitize the output name to be a valid filename
    sanitized_output_name = output_name.replace('/', '_').replace(' ', '_')
    
    shap.summary_plot(
        shap_values_by_output[i],
        X_explain_df.values,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(f'SHAP Feature Importance for {output_name}')
    plt.tight_layout()
    plot_path = os.path.join(SHAP_PLOTS_DIR, f'shap_summary_{sanitized_output_name}.png')
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

print("SHAP analysis complete.")