#!/usr/bin/env python3
"""
Clinician Policy SHAP Analysis
=============================

This script analyzes the clinician Bayesian policy model using SHAP explanations
to understand how patient state features influence clinical decision-making.

The script provides:
- Feature importance analysis across all treatments
- Treatment-specific decision patterns
- Comparative analysis of clinical priorities
- Comprehensive visualizations and insights

Usage:
    python clinician_shap_analysis.py --n_samples 1000 --n_background 100
    python clinician_shap_analysis.py --monte_carlo_samples 10
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import warnings
import shap
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add ALPACA path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
alpaca_path = os.path.join(parent_dir, 'reinforcement_learning', 'ALPACA')
sys.path.append(alpaca_path)

from alpaca_env import ALPACAEnv


# Bayesian model classes (copied from clinician_bayesian_model.py)
class CustomBayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=2.0, scale_factor=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_scale_factor = nn.Parameter(torch.tensor(np.log(scale_factor), dtype=torch.float32))
        
        # Variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.weight_mu, 0, 0.01)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.weight_rho, -4)
        nn.init.constant_(self.bias_rho, -4)
        
    def get_sigma(self, rho):
        return 0.05 * F.softplus(rho)
        
    def forward(self, x):
        weight_sigma = self.get_sigma(self.weight_rho)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        
        bias_sigma = self.get_sigma(self.bias_rho)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return F.linear(x, weight, bias)
        
    def kl_loss(self):
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


class BayesianModel(nn.Module):
    def __init__(self, input_size, num_continuous_outputs, num_binary_outputs, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_continuous_outputs = num_continuous_outputs
        self.num_binary_outputs = num_binary_outputs
        self.layer_size = 512
        
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
        
        self.fc3 = CustomBayesianLinear(self.layer_size, self.layer_size)
        self.act3 = nn.ELU()
        self.ln3 = nn.LayerNorm(self.layer_size)
        self.dropout3 = nn.Dropout(p=dropout_rate)

        # Output layers
        if self.num_continuous_outputs > 0:
            self.continuous_output = CustomBayesianLinear(self.layer_size, self.num_continuous_outputs, scale_factor=0.005)
        
        if self.num_binary_outputs > 0:
            self.binary_output = CustomBayesianLinear(self.layer_size, self.num_binary_outputs, scale_factor=0.005)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)

            if self.num_binary_outputs > 0:
                if self.num_continuous_outputs > 0:
                    continuous_mean, binary_logits = outputs
                    binary_probs = torch.sigmoid(binary_logits)
                    return continuous_mean, binary_probs
                else:  # Only binary outputs
                    _, binary_logits = outputs
                    binary_probs = torch.sigmoid(binary_logits)
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
                return torch.empty(x.size(0), 0, device=x.device), binary_logits
        else:
            if self.num_continuous_outputs > 0:
                return self.continuous_output(x)
            else:
                raise ValueError("The model has no output layers defined.")


class ClinicianShapAnalyzer:
    """SHAP analyzer for the clinician Bayesian policy model"""
    
    def __init__(self, model_path: str = None, scaler_path: str = None, monte_carlo_samples: int = 5):
        """
        Initialize the clinician SHAP analyzer
        
        Args:
            model_path: Path to the trained clinician model (.pth file)
            scaler_path: Path to the scaler (.joblib file)
            monte_carlo_samples: Number of MC samples for uncertainty estimation
        """
        self.monte_carlo_samples = monte_carlo_samples
        
        # Set default paths if not provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if model_path is None:
            model_path = os.path.join(parent_dir, 'treatment_comparison', 'best_clinician_policy.pth')
        if scaler_path is None:
            scaler_path = os.path.join(parent_dir, 'treatment_comparison', 'scaler_clinician_X.joblib')
            
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Initialize device
        self.device = torch.device('cpu')
        
        # Load the model and environment
        self.load_model()
        self.create_environment()
        self.setup_feature_names()
        
        print("Clinician Policy SHAP Analyzer initialized successfully")
        print(f"Model: {self.model_path}")
        print(f"Environment: ALPACA with {len(self.feature_names)} input features")
        print(f"Actions: {len(self.action_names)} treatment options")
        print(f"Monte Carlo samples: {self.monte_carlo_samples}")
    
    def load_model(self):
        """Load the trained clinician Bayesian model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Clinician model not found at {self.model_path}")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
        
        # Load the scaler and infer input feature ordering
        self.scaler = joblib.load(self.scaler_path)
        print(f"Loaded clinician scaler from {self.scaler_path}")

        self.feature_names = list(getattr(self.scaler, 'feature_names_in_', []))
        input_size = len(self.feature_names)
        if input_size == 0:
            raise ValueError("Clinician scaler missing feature_names_in_; cannot determine input size.")
        num_binary_outputs = 17
        num_continuous_outputs = 0
        
        self.model = BayesianModel(
            input_size=input_size,
            num_continuous_outputs=num_continuous_outputs,
            num_binary_outputs=num_binary_outputs
        ).to(self.device)
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded clinician model from {self.model_path}")
        print(f"Model expects {input_size} features (from scaler feature_names_in_)")
    
    def create_environment(self):
        """Create the ALPACA environment"""
        try:
            # Change to ALPACA directory to access data files
            original_dir = os.getcwd()
            os.chdir(alpaca_path)
            
            # Create environment
            self.env = ALPACAEnv(data_path='.', force_baseline_start=False)
            
            # Return to original directory
            os.chdir(original_dir)
            
            print("ALPACA environment created successfully")
            
        except Exception as e:
            print(f"Error creating ALPACA environment: {e}")
            raise
    
    def setup_feature_names(self):
        """Set up human-readable feature names"""
        # For clinician SHAP, use clinician state features (from scaler ordering)
        # Action names are sourced from the environment
        # self.feature_names already set in load_model
        self.action_names = self.env.action_cols.copy()
        
        # Create more readable feature names
        self.readable_feature_names = {
            'TAU_data': 'Tau Protein Level',
            'subject_age': 'Patient Age',
            'ABETA': 'Amyloid Beta Level', 
            'TRABSCOR': 'Cognitive Score (TRABSCOR)',
            'Ventricles': 'Ventricular Volume',
            'Hippocampus': 'Hippocampal Volume',
            'WholeBrain': 'Whole Brain Volume',
            'Entorhinal': 'Entorhinal Cortex Volume',
            'Fusiform': 'Fusiform Gyrus Volume',
            'MidTemp': 'Middle Temporal Volume',
            'ICV': 'Intracranial Volume',
            'months_since_bl': 'Months Since Baseline',
            'PTGENDER_Male': 'Male',
            'PTGENDER_Female': 'Female',
            'PTRACCAT_Asian': 'Asian Ethnicity',
            'PTRACCAT_Black': 'Black Ethnicity', 
            'PTRACCAT_Hawaiian/Other PI': 'Hawaiian/Pacific Islander',
            'PTRACCAT_More than one': 'Mixed Ethnicity',
            'PTRACCAT_White': 'White Ethnicity',
            'PTMARRY_Divorced': 'Divorced',
            'PTMARRY_Married': 'Married',
            'PTMARRY_Never married': 'Never Married',
            'PTMARRY_Unknown': 'Unknown Marital Status',
            'PTMARRY_Widowed': 'Widowed'
        }
        
        # Create readable action names
        self.readable_action_names = {
            'AD Treatment_active': 'AD Treatment',
            'Alpha Blocker_active': 'Alpha Blocker',
            'Analgesic_active': 'Pain Medication',
            'Antidepressant_active': 'Antidepressant',
            'Antihypertensive_active': 'Blood Pressure Med',
            'Bone Health_active': 'Bone Health Med',
            'Diabetes Medication_active': 'Diabetes Med',
            'Diuretic_active': 'Diuretic',
            'NSAID_active': 'Anti-inflammatory',
            'No Medication_active': 'No Medication',
            'Other_active': 'Other Medication',
            'PPI_active': 'Acid Reducer',
            'SSRI_active': 'SSRI Antidepressant',
            'Statin_active': 'Cholesterol Med',
            'Steroid_active': 'Steroid',
            'Supplement_active': 'Supplement',
            'Thyroid Hormone_active': 'Thyroid Hormone'
        }
    
    def _assemble_input(self, state: np.ndarray) -> np.ndarray:
        """Map ALPACA observation into clinician feature order and apply scaler normalization."""
        obs_idx = {name: i for i, name in enumerate(getattr(self.env, 'observation_cols', []))}
        row = {}
        for name in self.feature_names:
            if name in obs_idx:
                row[name] = float(state[obs_idx[name]])
            elif name in ('months_since_bl', 'MonthsSinceBL'):
                row[name] = float(getattr(self.env, '_months_since_bl', 0.0))
            elif name in ('next_visit_months', 'time_delta', 'time_since_prev'):
                row[name] = float(getattr(self.env, 'time_delta_val', 0.0))
            else:
                row[name] = 0.0
        import pandas as pd
        df = pd.DataFrame([[row[n] for n in self.feature_names]], columns=self.feature_names)
        if hasattr(self.env, 'manage_state_scaling'):
            df_scaled = self.env.manage_state_scaling(df, self.scaler, normalize=True)
        else:
            df_scaled = df
        return df_scaled.values.astype(np.float32)[0]

    def policy_function(self, observations):
        """
        Policy function that returns action probabilities for SHAP analysis
        
        Args:
            observations (np.ndarray): Array of observations [n_samples, n_features]
            
        Returns:
            np.ndarray: Action probabilities [n_samples, n_actions]
        """
        # Handle both single observation and batch of observations
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        
        action_probs_list = []
        
        for obs in observations:
            # Assemble clinician input features from env observation
            features = self._assemble_input(obs)
            obs_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32).to(self.device)
            
            # Use Monte Carlo sampling for more stable predictions
            mc_probs = []
            for _ in range(self.monte_carlo_samples):
                # Get predictions from Bayesian model
                with torch.no_grad():
                    _, binary_probs = self.model.predict(obs_tensor)
                    mc_probs.append(binary_probs.cpu().numpy())
            
            # Average the Monte Carlo samples
            action_probs = np.mean(mc_probs, axis=0).flatten()
            action_probs_list.append(action_probs)
        
        return np.array(action_probs_list)
    
    def analyze_action_distribution(self, n_test_samples=100):
        """Analyze the action distribution to understand model behavior"""
        print("Analyzing clinician action distribution...")
        
        # Generate some test observations
        test_obs = []
        for _ in range(n_test_samples):
            obs, _ = self.env.reset()
            test_obs.append(obs)
        
        test_obs = np.array(test_obs)
        
        # Get action probabilities
        action_probs = self.policy_function(test_obs)
        
        # Analyze the distribution
        mean_probs = np.mean(action_probs, axis=0)
        std_probs = np.std(action_probs, axis=0)
        min_probs = np.min(action_probs, axis=0)
        max_probs = np.max(action_probs, axis=0)
        
        print(f"Action Probability Statistics ({n_test_samples} samples):")
        print(f"   Mean: {mean_probs}")
        print(f"   Std:  {std_probs}")
        print(f"   Min:  {min_probs}")
        print(f"   Max:  {max_probs}")
        
        return {
            'mean': mean_probs,
            'std': std_probs,
            'min': min_probs,
            'max': max_probs,
            'action_names': self.action_names
        }
    
    def generate_sample_data(self, n_samples=1000):
        """
        Generate sample data from the environment for SHAP analysis
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            np.ndarray: Sample observations [n_samples, n_features]
        """
        print(f"Generating {n_samples} sample observations for clinician analysis...")
        
        observations = []
        
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"  Generated {i}/{n_samples} samples...")
                
            # Reset environment to get a new initial state
            obs, _ = self.env.reset()
            observations.append(obs)
            
            # Optionally take a few random steps to get diverse states
            n_steps = np.random.randint(0, 5)  # 0-4 additional steps
            for _ in range(n_steps):
                action = self.env.action_space.sample()
                obs, _, done, truncated, _ = self.env.step(action)
                if done or truncated:
                    break
        
        sample_data = np.array(observations)
        print(f"Generated {len(sample_data)} sample observations")
        print(f"Sample shape: {sample_data.shape}")
        
        return sample_data
    
    def run_shap_analysis(self, sample_data, n_background=100, save_results=True, fast_mode=False):
        """
        Run SHAP analysis on the clinician policy
        
        Args:
            sample_data (np.ndarray): Sample observations for analysis
            n_background (int): Number of background samples for SHAP explainer
            save_results (bool): Whether to save results to files
            fast_mode (bool): Whether to use fast mode optimizations
            
        Returns:
            dict: SHAP analysis results
        """
        print("Running SHAP analysis for clinician policy...")
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Background samples: {n_background}")
        
        # Analyze action distribution first (fewer samples in fast mode)
        test_samples = 25 if fast_mode else 50
        action_stats = self.analyze_action_distribution(n_test_samples=test_samples)
        
        # Select background data (subset of sample data)
        background_indices = np.random.choice(len(sample_data), size=min(n_background, len(sample_data)), replace=False)
        background_data = sample_data[background_indices].astype(np.float32)
        
        # Create SHAP explainer
        print("Creating SHAP explainer (Explainer)...")
        explainer = shap.Explainer(self.policy_function, background_data)
        
        # Calculate SHAP values for a subset of data (fewer in fast mode)
        if fast_mode:
            n_explain = min(50, len(sample_data))  # Much fewer samples for fast mode
        else:
            n_explain = min(200, len(sample_data))
        explain_indices = np.random.choice(len(sample_data), size=n_explain, replace=False)
        explain_data = sample_data[explain_indices].astype(np.float32)
        
        print(f"Calculating SHAP values for {n_explain} samples...")
        shap_values = explainer(explain_data)
        print("SHAP analysis completed for clinician policy")
        print(f"SHAP values shape: {shap_values.values.shape}")

        # Additivity check similar to RL analyzer
        additivity_mae = None
        try:
            preds = self.policy_function(explain_data)
            base_vals = np.array(shap_values.base_values)
            contribs = shap_values.values.sum(axis=1)
            if base_vals.ndim == 1 and preds.ndim == 2:
                base_vals = np.tile(base_vals.reshape(-1, 1), (1, preds.shape[1]))
            recon = base_vals + contribs
            additivity_mae = float(np.mean(np.abs(recon - preds)))
            print(f"SHAP additivity check MAE (reconstruction vs. policy outputs): {additivity_mae:.6f}")
        except Exception as _e:
            print(f"Skipped additivity check due to: {_e}")
        
        # Create results dictionary
        results = {
            'algorithm': 'clinician_bayesian',
            'model_path': self.model_path,
            'shap_values': shap_values,
            'sample_data': sample_data,
            'explain_data': explain_data,
            'background_data': background_data,
            'feature_names': self.feature_names,
            'action_names': self.action_names,
            'readable_feature_names': self.readable_feature_names,
            'readable_action_names': self.readable_action_names,
            'action_stats': action_stats,
            'additivity_mae': additivity_mae,
        }
        
        # Save results if requested
        if save_results:
            results_dir = self.save_shap_results(results)
            
            # Create visualizations (fewer in fast mode)
            viz_dir = self.create_visualizations(results, results_dir, fast_mode=fast_mode)
            results['visualization_dir'] = viz_dir
        
        return results
    
    def save_shap_results(self, results):
        """Save SHAP results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"clinician_shap_analysis_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Saving clinician SHAP results to {results_dir}/")
        
        # Save SHAP values as numpy arrays
        np.save(f"{results_dir}/shap_values.npy", results['shap_values'].values)
        np.save(f"{results_dir}/sample_data.npy", results['sample_data'])
        np.save(f"{results_dir}/explain_data.npy", results['explain_data'])
        
        # Save metadata
        metadata = {
            'algorithm': results['algorithm'],
            'model_path': results['model_path'],
            'timestamp': timestamp,
            'n_samples': len(results['sample_data']),
            'n_explain': len(results['explain_data']),
            'n_features': len(results['feature_names']),
            'n_actions': len(results['action_names']),
            'monte_carlo_samples': self.monte_carlo_samples,
            'additivity_mae': results.get('additivity_mae'),
        }
        pd.Series(metadata).to_csv(f"{results_dir}/metadata.csv")
        
        # Save feature and action names
        pd.Series(results['feature_names']).to_csv(f"{results_dir}/feature_names.csv", index=False)
        pd.Series(results['action_names']).to_csv(f"{results_dir}/action_names.csv", index=False)
        
        # Save readable names
        pd.DataFrame(list(results['readable_feature_names'].items()), 
                    columns=['original', 'readable']).to_csv(f"{results_dir}/readable_feature_names.csv", index=False)
        pd.DataFrame(list(results['readable_action_names'].items()), 
                    columns=['original', 'readable']).to_csv(f"{results_dir}/readable_action_names.csv", index=False)
        
        # Save action statistics
        action_stats_df = pd.DataFrame(results['action_stats'])
        action_stats_df.to_csv(f"{results_dir}/action_statistics.csv", index=False)
        
        print(f"Results saved to {results_dir}/")
        return results_dir
    
    def create_visualizations(self, results, output_dir, fast_mode=False):
        """Create comprehensive visualizations for clinician analysis"""
        viz_dir = output_dir.replace('shap_analysis_', 'shap_visualizations_')
        os.makedirs(viz_dir, exist_ok=True)
        
        mode_str = " (Fast Mode)" if fast_mode else ""
        print(f"Creating clinician visualizations{mode_str} in {viz_dir}/")
        
        shap_values = results['shap_values']
        feature_names = results['feature_names']
        action_names = results['action_names']
        readable_feature_names = results['readable_feature_names']
        readable_action_names = results['readable_action_names']
        
        # 1. Overall feature importance
        plt.figure(figsize=(12, 10))
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 2))
        readable_features = [readable_feature_names.get(f, f) for f in feature_names]
        
        importance_df = pd.DataFrame({
            'feature': readable_features,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(importance_df)), importance_df['importance'], 
                color='#2E86AB', alpha=0.7)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Clinician Policy - Overall Feature Importance', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/overall_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Action-specific importance heatmap
        plt.figure(figsize=(16, 12))
        
        # Calculate action-specific importances
        action_importance_matrix = []
        readable_actions = []
        
        for action_idx, action_name in enumerate(action_names):
            action_shap = shap_values.values[:, :, action_idx]
            action_importance = np.mean(np.abs(action_shap), axis=0)
            action_importance_matrix.append(action_importance)
            readable_actions.append(readable_action_names.get(action_name, action_name))
        
        action_importance_matrix = np.array(action_importance_matrix)
        
        # Create heatmap
        sns.heatmap(action_importance_matrix, 
                   xticklabels=readable_features,
                   yticklabels=readable_actions,
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Mean |SHAP Value|'})
        
        plt.title('Clinician Policy - Action-Specific Feature Importance', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Patient Features')
        plt.ylabel('Treatment Options')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/action_specific_importance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top 10 features bar chart
        plt.figure(figsize=(10, 8))
        top_10 = importance_df.tail(10)
        
        plt.barh(range(len(top_10)), top_10['importance'], 
                color='#2E86AB', alpha=0.8)
        plt.yticks(range(len(top_10)), top_10['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Clinician Policy - Top 10 Most Important Features', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/top_10_features.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Clinical decision patterns for key treatments (skip in fast mode)
        if not fast_mode:
            key_treatments = ['AD Treatment_active', 'Antidepressant_active', 'Statin_active', 
                             'No Medication_active', 'Antihypertensive_active', 'NSAID_active']
            
            available_treatments = [t for t in key_treatments if t in action_names]
            
            if available_treatments:
                n_treatments = len(available_treatments)
                cols = min(3, n_treatments)
                rows = (n_treatments + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                if n_treatments == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes if n_treatments > 1 else [axes]
                else:
                    axes = axes.flatten()
                
                for i, treatment in enumerate(available_treatments):
                    if i >= len(axes):
                        break
                        
                    action_idx = action_names.index(treatment)
                    action_shap = shap_values.values[:, :, action_idx]
                    action_importance = np.mean(np.abs(action_shap), axis=0)
                    
                    # Get top features for this treatment
                    top_features_idx = np.argsort(action_importance)[-8:]
                    top_features_names = [readable_features[idx] for idx in top_features_idx]
                    top_features_values = action_importance[top_features_idx]
                    
                    axes[i].barh(range(len(top_features_values)), top_features_values,
                               color='#A23B72', alpha=0.7)
                    axes[i].set_yticks(range(len(top_features_names)))
                    axes[i].set_yticklabels(top_features_names, fontsize=9)
                    axes[i].set_xlabel('Mean |SHAP Value|')
                    axes[i].set_title(f'{readable_action_names.get(treatment, treatment)}',
                                    fontweight='bold')
                    axes[i].grid(axis='x', alpha=0.3)
                
                # Hide unused subplots
                for i in range(len(available_treatments), len(axes)):
                    axes[i].set_visible(False)
                
                plt.suptitle('Clinician Policy - Key Treatment Decision Patterns', 
                            fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/key_treatment_patterns.png", dpi=300, bbox_inches='tight')
                plt.close()
        else:
            print("Skipping detailed treatment patterns in fast mode")
        
        # 5. Feature category analysis (skip in fast mode for speed)
        if not fast_mode:
            self.create_feature_category_analysis(results, viz_dir)
        else:
            print("Skipping feature category analysis in fast mode")
        
        # 6. Clinical insights summary (always create)
        self.create_clinical_insights_summary(results, viz_dir)
        
        print(f"Visualizations saved to {viz_dir}/")
        return viz_dir
    
    def create_feature_category_analysis(self, results, viz_dir):
        """Create analysis by feature categories (biomarkers, demographics, etc.)"""
        shap_values = results['shap_values']
        feature_names = results['feature_names']
        readable_feature_names = results['readable_feature_names']
        
        # Define feature categories
        feature_categories = {
            'Biomarkers': ['TAU_data', 'ABETA', 'TRABSCOR'],
            'Brain Volumes': ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV'],
            'Demographics': ['subject_age', 'PTGENDER_Male'],
            'Ethnicity': ['PTRACCAT_Asian', 'PTRACCAT_Black', 'PTRACCAT_Hawaiian/Other PI', 
                         'PTRACCAT_More than one', 'PTRACCAT_White'],
            'Marital Status': ['PTMARRY_Divorced', 'PTMARRY_Married', 'PTMARRY_Never married', 
                              'PTMARRY_Unknown', 'PTMARRY_Widowed'],
            'Temporal': ['months_since_bl']
        }
        
        # Calculate category importances
        category_importances = {}
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 2))
        
        for category, features in feature_categories.items():
            category_importance = 0
            feature_count = 0
            for feature in features:
                if feature in feature_names:
                    feature_idx = feature_names.index(feature)
                    category_importance += mean_abs_shap[feature_idx]
                    feature_count += 1
            if feature_count > 0:
                category_importances[category] = category_importance / feature_count
        
        # Create category importance plot
        plt.figure(figsize=(10, 6))
        categories = list(category_importances.keys())
        importances = list(category_importances.values())
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
        
        plt.bar(categories, importances, color=colors[:len(categories)], alpha=0.8)
        plt.xlabel('Feature Category')
        plt.ylabel('Average Feature Importance')
        plt.title('Clinician Decision-Making by Feature Category', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/feature_category_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_clinical_insights_summary(self, results, viz_dir):
        """Create a clinical insights summary visualization"""
        shap_values = results['shap_values']
        feature_names = results['feature_names']
        action_names = results['action_names']
        readable_feature_names = results['readable_feature_names']
        readable_action_names = results['readable_action_names']
        action_stats = results['action_stats']
        
        # Calculate key insights
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 2))
        top_feature_idx = np.argmax(mean_abs_shap)
        top_feature = readable_feature_names.get(feature_names[top_feature_idx], feature_names[top_feature_idx])
        
        # Find most commonly prescribed treatments
        most_prescribed_idx = np.argmax(action_stats['mean'])
        most_prescribed = readable_action_names.get(action_names[most_prescribed_idx], action_names[most_prescribed_idx])
        
        # Find treatments with highest variance (most conditional)
        most_conditional_idx = np.argmax(action_stats['std'])
        most_conditional = readable_action_names.get(action_names[most_conditional_idx], action_names[most_conditional_idx])
        
        # Create summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top features
        top_10_features = np.argsort(mean_abs_shap)[-10:]
        top_10_names = [readable_feature_names.get(feature_names[i], feature_names[i]) for i in top_10_features]
        top_10_values = mean_abs_shap[top_10_features]
        
        ax1.barh(range(len(top_10_values)), top_10_values, color='#2E86AB', alpha=0.7)
        ax1.set_yticks(range(len(top_10_names)))
        ax1.set_yticklabels(top_10_names)
        ax1.set_xlabel('Mean |SHAP Value|')
        ax1.set_title('Top 10 Clinical Decision Factors', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Treatment prescription rates
        readable_actions = [readable_action_names.get(name, name) for name in action_names]
        ax2.bar(range(len(action_stats['mean'])), action_stats['mean'], 
               color='#A23B72', alpha=0.7)
        ax2.set_xlabel('Treatment Options')
        ax2.set_ylabel('Prescription Probability')
        ax2.set_title('Average Treatment Prescription Rates', fontweight='bold')
        ax2.set_xticks(range(len(readable_actions)))
        ax2.set_xticklabels(readable_actions, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Treatment variability
        ax3.bar(range(len(action_stats['std'])), action_stats['std'], 
               color='#F18F01', alpha=0.7)
        ax3.set_xlabel('Treatment Options')
        ax3.set_ylabel('Prescription Variability (Std)')
        ax3.set_title('Treatment Decision Variability', fontweight='bold')
        ax3.set_xticks(range(len(readable_actions)))
        ax3.set_xticklabels(readable_actions, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Key insights text
        insights_text = f"""
        CLINICAL DECISION-MAKING INSIGHTS

        Most Important Factor: {top_feature}
        Most Prescribed Treatment: {most_prescribed}
        Most Conditional Treatment: {most_conditional}

        Summary:
        - Total Samples: {len(results['sample_data'])}
        - Features Analyzed: {len(feature_names)}
        - Treatments Modeled: {len(action_names)}
        - Monte Carlo Samples: {self.monte_carlo_samples}

        Key Findings:
        - Clinician decisions strongly depend on {top_feature.lower()}
        - {most_prescribed} is the most commonly prescribed treatment
        - {most_conditional} shows highest patient-specific variability
        - Bayesian uncertainty captured through MC sampling
        """
        
        ax4.text(0.05, 0.95, insights_text, 
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Clinical Insights Summary', fontweight='bold')
        
        plt.suptitle('Clinician Policy SHAP Analysis - Comprehensive Summary', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/clinical_insights_summary.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the clinician SHAP analysis"""
    parser = argparse.ArgumentParser(description='SHAP Analysis for Clinician Bayesian Policy')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained clinician model (.pth file)')
    parser.add_argument('--scaler_path', type=str, default=None,
                       help='Path to scaler (.joblib file)')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of samples to generate for analysis (default: 1000)')
    parser.add_argument('--n_background', type=int, default=50,
                       help='Number of background samples for SHAP explainer (default: 100)')
    parser.add_argument('--monte_carlo_samples', type=int, default=5,
                       help='Number of Monte Carlo samples for uncertainty (default: 5)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: reduced samples for ~15min runtime (overrides other sample arguments)')
    
    args = parser.parse_args()
    
    # Apply fast mode settings if requested
    if args.fast:
        print("FAST MODE ENABLED - Optimized for ~15 minute runtime")
        args.n_samples = 200          # Reduced from 1000
        args.n_background = 30        # Reduced from 100  
        args.monte_carlo_samples = 5  # Reduced from 5
        print(f"   Samples: {args.n_samples}")
        print(f"   Background: {args.n_background}")
        print(f"   MC Samples: {args.monte_carlo_samples}")
        print()
    
    print("Clinician Policy SHAP Analysis")
    print("=" * 80)
    
    try:
        # Create analyzer
        analyzer = ClinicianShapAnalyzer(
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            monte_carlo_samples=args.monte_carlo_samples
        )
        
        # Generate sample data
        sample_data = analyzer.generate_sample_data(n_samples=args.n_samples)
        
        # Run SHAP analysis
        results = analyzer.run_shap_analysis(
            sample_data, 
            n_background=args.n_background,
            fast_mode=args.fast
        )
        
        print(f"\n Clinician Policy SHAP Analysis completed successfully!")
        print(f" Results saved to: {results.get('visualization_dir', 'results directory')}")
        
        return True
        
    except Exception as e:
        print(f" SHAP Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
