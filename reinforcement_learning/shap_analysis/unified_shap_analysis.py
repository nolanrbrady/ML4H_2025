#!/usr/bin/env python3
"""
Unified SHAP Analysis for Reinforcement Learning Models
======================================================

This script provides a unified interface for analyzing PPO, A2C, and SAC models
trained on the ALPACA environment using SHAP explanations.

Features:
- Automatically detects and analyzes models from all three algorithms
- Comparative analysis across algorithms
- Algorithm-specific configurations
- Easy paper-ready outputs and visualizations

Usage:
    python unified_shap_analysis.py --algorithm ppo
    python unified_shap_analysis.py --algorithm a2c
    python unified_shap_analysis.py --algorithm sac
    python unified_shap_analysis.py --algorithm all --compare
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import warnings
import shap
warnings.filterwarnings('ignore')


# Add ALPACA path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'ALPACA'))

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import VecNormalize
from alpaca_env import ALPACAEnv
import joblib


class UnifiedShapAnalyzer:
    """Unified SHAP analyzer for multiple RL algorithms on ALPACA environment"""
    
    ALGORITHM_CONFIGS = {
        'ppo': {
            'model_class': PPO,
            'pattern': 'ppo_alpaca_*',
            'name': 'PPO (Proximal Policy Optimization)',
            'color': '#1f77b4'
        },
        'a2c': {
            'model_class': A2C,
            'pattern': 'a2c_alpaca_*',
            'name': 'A2C (Advantage Actor-Critic)',
            'color': '#ff7f0e'
        },
        'sac': {
            'model_class': SAC,
            'pattern': 'sac_alpaca_stable_*',
            'name': 'SAC (Soft Actor-Critic)',
            'color': '#2ca02c'
        }
    }
    
    def __init__(self, algorithm, model_dir=None, vec_normalize_path=None):
        """
        Initialize the unified SHAP analyzer
        
        Args:
            algorithm (str): Algorithm type ('ppo', 'a2c', 'sac')
            model_dir (str): Path to the trained model directory (optional)
            vec_normalize_path (str): Path to VecNormalize object (optional)
        """
        self.algorithm = algorithm.lower()
        self.model_dir = model_dir
        self.vec_normalize_path = vec_normalize_path
        
        if self.algorithm not in self.ALGORITHM_CONFIGS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {list(self.ALGORITHM_CONFIGS.keys())}")
        
        self.config = self.ALGORITHM_CONFIGS[self.algorithm]
        
        # Find model directory if not provided
        if self.model_dir is None:
            self.model_dir = self.find_latest_model()
        
        # Reorder initialization to ensure environment exists for VecNormalize
        self.create_environment()
        self.load_model()
        self.setup_feature_names()
        
        print(f"SHAP Analyzer initialized successfully for {self.config['name']}")
        print(f"Model Directory: {self.model_dir}")
        print(f"Environment: ALPACA with {len(self.feature_names)} features and {len(self.action_names)} actions")
    
    def find_latest_model(self):
        """Find the latest model directory for the specified algorithm"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pattern = os.path.join(parent_dir, self.config['pattern'])
        model_dirs = glob.glob(pattern)
        
        if not model_dirs:
            raise FileNotFoundError(f"No {self.algorithm.upper()} model directories found matching pattern: {self.config['pattern']}")
        
        model_dirs.sort(reverse=True)
        latest_model = model_dirs[0]
        
        print(f"Found {len(model_dirs)} {self.algorithm.upper()} model directories. Latest: {os.path.basename(latest_model)}")
        return latest_model
    
    def load_model(self):
        """Load the trained model and VecNormalize object"""
        best_model_path = os.path.join(self.model_dir, 'best_model', 'best_model.zip')
        final_model_path = os.path.join(self.model_dir, 'final_model.zip')
        
        model_path = best_model_path if os.path.exists(best_model_path) else final_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found in {self.model_dir}")
            
        self.model = self.config['model_class'].load(model_path)
        print(f"Loaded {self.algorithm.upper()} model from: {model_path}")
        
        # Load VecNormalize if available
        if self.vec_normalize_path is None:
            self.vec_normalize_path = os.path.join(self.model_dir, 'vec_normalize.pkl')
        
        if os.path.exists(self.vec_normalize_path):
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv
                dummy_env = DummyVecEnv([lambda: self.env])
                self.vec_normalize = VecNormalize.load(self.vec_normalize_path, dummy_env)
                self.vec_normalize.venv = None  # Decouple from the dummy env
                print(f"Loaded VecNormalize from {self.vec_normalize_path}")
            except Exception as e:
                print(f"Warning: Could not load VecNormalize: {e}. Proceeding without normalization.")
                self.vec_normalize = None
        else:
            self.vec_normalize = None
            print("No VecNormalize found, proceeding with raw observations.")
    
    def create_environment(self):
        """Create the ALPACA environment"""
        try:
            # Change to ALPACA directory to access data files
            original_dir = os.getcwd()
            alpaca_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ALPACA')
            os.chdir(alpaca_dir)
            
            # Create environment
            self.env = ALPACAEnv(data_path='.', force_baseline_start=False)
            
            # Return to original directory
            os.chdir(original_dir)
            
            print(f"ALPACA environment created successfully")
            
        except Exception as e:
            print(f"Error creating ALPACA environment: {e}")
            raise
    
    def setup_feature_names(self):
        """Set up human-readable feature names"""
        self.feature_names = self.env.observation_cols.copy()
        self.action_names = self.env.action_cols.copy()
        
        # Create more readable feature names
        self.readable_feature_names = {
            'ADNI_MEM': 'Memory (ADNI_MEM)',
            'ADNI_EF2': 'Executive Function (ADNI_EF2)',
            'TAU_data': 'Tau Protein Level',
            'subject_age': 'Patient Age',
            'ABETA': 'Amyloid Beta Level', 
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
            'PTRACCAT_Am Indian/Alaskan': 'American Indian/Alaskan Native',
            'PTRACCAT_Unknown': 'Unknown Ethnicity',
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
            # Apply VecNormalize if available
            if self.vec_normalize is not None:
                try:
                    # VecNormalize expects observations in the format it was trained with
                    normalized_obs = self.vec_normalize.normalize_obs(obs.reshape(1, -1))
                    obs_tensor = torch.tensor(normalized_obs, dtype=torch.float32)
                except Exception as e:
                    # Fall back to raw observations if normalization fails
                    obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
            else:
                obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
            
            # Get action distribution from policy
            with torch.no_grad():
                if self.algorithm == 'sac':
                    # SAC has continuous actions but we need action probabilities
                    # Sample multiple actions to capture the stochastic policy properly
                    n_samples = 10  # Reduced from 20 for efficiency, but enough for good estimates
                    sampled_probs = []
                    
                    for _ in range(n_samples):
                        # Sample action stochastically
                        action, _ = self.model.predict(obs_tensor.numpy(), deterministic=False)
                        
                        # Convert continuous actions to binary probabilities
                        continuous_actions = action.flatten()
                        
                        # Check if we should use enhanced conversion for better differentiation
                        if hasattr(self, '_use_enhanced_sac_conversion') and self._use_enhanced_sac_conversion:
                            # Enhanced method: Use softmax to create more differentiated probabilities
                            # Scale up the continuous actions to amplify differences
                            scaled_actions = continuous_actions * 2.0  # Amplify differences
                            probs = torch.softmax(torch.tensor(scaled_actions), dim=0).numpy()
                        else:
                            # Standard method: Direct scaling
                            probs = (continuous_actions + 1.0) / 2.0  # Scale [-1,1] to [0,1]
                        
                        # Ensure we have the right number of actions
                        if len(probs) != len(self.action_names):
                            if len(probs) < len(self.action_names):
                                probs = np.pad(probs, (0, len(self.action_names) - len(probs)), constant_values=0.5)
                            else:
                                probs = probs[:len(self.action_names)]
                        
                        sampled_probs.append(probs)
                    
                    # Average the sampled probabilities to get stable estimates
                    action_probs = np.mean(sampled_probs, axis=0)
                    
                    # Ensure probabilities are valid and have meaningful variance
                    if hasattr(self, '_use_enhanced_sac_conversion') and self._use_enhanced_sac_conversion:
                        # For enhanced mode, allow wider range for better differentiation
                        action_probs = np.clip(action_probs, 0.01, 0.99)
                    else:
                        action_probs = np.clip(action_probs, 0.05, 0.95)
                else:
                    # PPO and A2C (MultiBinary Bernoulli per action)
                    # Ensure tensor on correct device
                    obs_tensor = obs_tensor.to(self.model.device)
                    dist = self.model.policy.get_distribution(obs_tensor)

                    # SB3 wraps torch distributions. Handle both Categorical and Independent(Bernoulli)
                    base_dist = getattr(dist.distribution, 'base_dist', dist.distribution)

                    probs_tensor = None
                    if hasattr(base_dist, 'probs') and base_dist.probs is not None:
                        probs_tensor = base_dist.probs
                    elif hasattr(base_dist, 'logits') and base_dist.logits is not None:
                        probs_tensor = torch.sigmoid(base_dist.logits)
                    else:
                        # Fallback via sampling multiple times to estimate marginal probs
                        samples = []
                        for _ in range(16):
                            act, _ = self.model.predict(obs_tensor.cpu().numpy(), deterministic=False)
                            samples.append(act.flatten())
                        action_probs = np.clip(np.mean(samples, axis=0), 0.001, 0.999)
                        action_probs_list.append(action_probs)
                        continue

                    # probs_tensor shape: (batch, n_actions) or (n_actions,)
                    probs = probs_tensor.detach().cpu().numpy()
                    if probs.ndim == 2:
                        action_probs = probs[0]
                    else:
                        action_probs = probs.flatten()
                
                action_probs_list.append(action_probs)
        
        return np.array(action_probs_list)
    
    def analyze_action_distribution(self, n_test_samples=100):
        """Analyze the action distribution to understand model behavior"""
        print(f"Analyzing {self.algorithm.upper()} action distribution...")
        
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
        
        # Check for differentiation
        if np.all(std_probs < 0.01):
            print("WARNING: Very low variance in action probabilities - model may not be well differentiated")
        
        if np.all(np.abs(mean_probs - 0.5) < 0.1):
            print("WARNING: All action probabilities near 0.5 - model may not have strong preferences")
        
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
        print(f"Generating {n_samples} sample observations for {self.algorithm.upper()}...")
        
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
    
    def run_shap_analysis(self, sample_data, n_background=100, save_results=True):
        """
        Run SHAP analysis on the policy
        
        Args:
            sample_data (np.ndarray): Sample observations for analysis
            n_background (int): Number of background samples for SHAP explainer
            save_results (bool): Whether to save results to files
            
        Returns:
            dict: SHAP analysis results
        """
        print(f"Running SHAP analysis for {self.algorithm.upper()}...")
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Background samples: {n_background}")
        
        # For SAC, analyze action distribution first to understand model behavior
        if self.algorithm == 'sac':
            action_stats = self.analyze_action_distribution(n_test_samples=50)
            
            # If the model shows poor differentiation, try alternative conversion method
            if np.all(action_stats['std'] < 0.02):
                print("Model shows low variance - applying enhanced differentiation...")
                self._use_enhanced_sac_conversion = True
            else:
                self._use_enhanced_sac_conversion = False
        
        # Select background data (subset of sample data)
        background_indices = np.random.choice(len(sample_data), size=min(n_background, len(sample_data)), replace=False)
        background_data = sample_data[background_indices].astype(np.float32)
        
        # Create SHAP explainer
        print(f"Creating SHAP explainer (KernelExplainer)...")
        # KernelExplainer is appropriate for black-box vector-output functions (action probabilities)
        explainer = shap.KernelExplainer(self.policy_function, background_data)
        
        # Calculate SHAP values for a subset of data (to avoid memory issues)
        n_explain = min(200, len(sample_data))
        explain_indices = np.random.choice(len(sample_data), size=n_explain, replace=False)
        explain_data = sample_data[explain_indices].astype(np.float32)
        
        print(f"Calculating SHAP values for {n_explain} samples...")
        shap_values = explainer(explain_data)
        
        print(f"SHAP analysis completed for {self.algorithm.upper()}")
        print(f"SHAP values shape: {shap_values.values.shape}")

        # Sanity checks: output dimension should match number of actions
        if shap_values.values.ndim == 3:
            n_outputs = shap_values.values.shape[2]
        elif shap_values.values.ndim == 2:
            # Degenerate case: single-output; expand dims for consistency
            n_outputs = 1
        else:
            n_outputs = None

        if n_outputs is not None and n_outputs != len(self.action_names):
            print(f"Warning: Model output dim ({n_outputs}) != actions ({len(self.action_names)})")

        # Validate additivity: base_value + sum(shap) ≈ model output
        try:
            preds = self.policy_function(explain_data)
            # shap base_values can be (n_samples, n_outputs)
            base_vals = np.array(shap_values.base_values)
            contribs = shap_values.values.sum(axis=1)
            if base_vals.ndim == 1 and preds.ndim == 2:
                base_vals = np.tile(base_vals.reshape(-1, 1), (1, preds.shape[1]))
            recon = base_vals + contribs
            recon_mae = np.mean(np.abs(recon - preds))
            print(f"SHAP additivity check MAE (reconstruction vs. policy outputs): {recon_mae:.6f}")
        except Exception as _e:
            print(f"Skipped additivity check due to: {_e}")
        
        # Create results dictionary
        results = {
            'algorithm': self.algorithm,
            'model_dir': self.model_dir,
            'shap_values': shap_values,
            'sample_data': sample_data,
            'explain_data': explain_data,
            'background_data': background_data,
            'additivity_mae': float(recon_mae) if 'recon_mae' in locals() else None,
            'feature_names': self.feature_names,
            'action_names': self.action_names,
            'readable_feature_names': self.readable_feature_names,
            'readable_action_names': self.readable_action_names,
            'config': self.config
        }
        
        # Save results if requested
        if save_results:
            results_dir = self.save_shap_results(results)
            
            # Create individual visualizations
            viz_dir = self.create_individual_visualizations(results, results_dir)
            results['visualization_dir'] = viz_dir
        
        return results
    
    def save_shap_results(self, results):
        """Save SHAP results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"{self.algorithm}_shap_analysis_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Saving {self.algorithm.upper()} SHAP results to {results_dir}/")
        
        # Save SHAP values as numpy arrays
        np.save(f"{results_dir}/shap_values.npy", results['shap_values'].values)
        np.save(f"{results_dir}/sample_data.npy", results['sample_data'])
        np.save(f"{results_dir}/explain_data.npy", results['explain_data'])
        
        # Save metadata
        metadata = {
            'algorithm': self.algorithm,
            'model_dir': self.model_dir,
            'timestamp': timestamp,
            'n_samples': len(results['sample_data']),
            'n_explain': len(results['explain_data']),
            'n_features': len(results['feature_names']),
            'n_actions': len(results['action_names'])
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
        
        print(f"Results saved to {results_dir}/")
        return results_dir
    
    def create_individual_visualizations(self, results, output_dir):
        """Create visualizations for individual algorithm analysis"""
        viz_dir = output_dir.replace('shap_analysis_', 'shap_visualizations_')
        os.makedirs(viz_dir, exist_ok=True)
        
        print(f"Creating {self.algorithm.upper()} visualizations in {viz_dir}/")
        
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
                color=self.config['color'], alpha=0.7)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'{self.config["name"]} - Overall Feature Importance', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/overall_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Action-specific importance heatmap
        plt.figure(figsize=(14, 10))
        
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
        
        plt.title(f'{self.config["name"]} - Action-Specific Feature Importance', 
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
                color=self.config['color'], alpha=0.8)
        plt.yticks(range(len(top_10)), top_10['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'{self.config["name"]} - Top 10 Most Important Features', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/top_10_features.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Treatment-specific analysis for meaningful actions
        meaningful_actions = []
        for action_idx, action_name in enumerate(action_names):
            action_shap = shap_values.values[:, :, action_idx]
            max_impact = np.max(np.abs(action_shap))
            if max_impact > 0.001:  # Threshold for meaningful impact
                meaningful_actions.append((action_idx, action_name, max_impact))
        
        if meaningful_actions:
            # Create subplot for each meaningful action
            n_actions = len(meaningful_actions)
            cols = min(3, n_actions)
            rows = (n_actions + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
            if n_actions == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if n_actions > 1 else [axes]
            else:
                axes = axes.flatten()
            
            for i, (action_idx, action_name, max_impact) in enumerate(meaningful_actions):
                if i >= len(axes):
                    break
                    
                action_shap = shap_values.values[:, :, action_idx]
                action_importance = np.mean(np.abs(action_shap), axis=0)
                
                # Get top features for this action
                top_features_idx = np.argsort(action_importance)[-8:]
                top_features_names = [readable_features[idx] for idx in top_features_idx]
                top_features_values = action_importance[top_features_idx]
                
                axes[i].barh(range(len(top_features_values)), top_features_values,
                           color=self.config['color'], alpha=0.7)
                axes[i].set_yticks(range(len(top_features_names)))
                axes[i].set_yticklabels(top_features_names, fontsize=9)
                axes[i].set_xlabel('Mean |SHAP Value|')
                axes[i].set_title(f'{readable_action_names.get(action_name, action_name)}',
                                fontweight='bold')
                axes[i].grid(axis='x', alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(meaningful_actions), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'{self.config["name"]} - Feature-Dependent Treatment Decisions', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/meaningful_treatments.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Summary statistics visualization
        plt.figure(figsize=(12, 8))
        
        # Create summary statistics
        summary_data = {
            'Total Samples': len(results['sample_data']),
            'Total Features': len(feature_names),
            'Total Actions': len(action_names),
            'Meaningful Actions': len(meaningful_actions),
            'Max SHAP Value': np.max(np.abs(shap_values.values)),
            'Mean SHAP Value': np.mean(np.abs(shap_values.values)),
            'Top Feature Score': np.max(mean_abs_shap)
        }
        
        # Create a summary text plot
        plt.text(0.05, 0.95, f'{self.config["name"]} Analysis Summary', 
                fontsize=20, fontweight='bold', transform=plt.gca().transAxes)
        
        y_pos = 0.8
        for key, value in summary_data.items():
            if isinstance(value, float):
                plt.text(0.05, y_pos, f'{key}: {value:.4f}', 
                        fontsize=14, transform=plt.gca().transAxes)
            else:
                plt.text(0.05, y_pos, f'{key}: {value}', 
                        fontsize=14, transform=plt.gca().transAxes)
            y_pos -= 0.08
        
        if meaningful_actions:
            plt.text(0.05, y_pos-0.02, 'Feature-Dependent Treatments:', 
                    fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
            y_pos -= 0.06
            for _, action_name, max_impact in meaningful_actions:
                readable_name = readable_action_names.get(action_name, action_name)
                plt.text(0.1, y_pos, f'• {readable_name} (max impact: {max_impact:.4f})', 
                        fontsize=12, transform=plt.gca().transAxes)
                y_pos -= 0.05
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/analysis_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual visualizations saved to {viz_dir}/")
        return viz_dir


def find_all_models():
    """Find all available models across algorithms"""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_models = {}
    
    for algorithm, config in UnifiedShapAnalyzer.ALGORITHM_CONFIGS.items():
        pattern = os.path.join(parent_dir, config['pattern'])
        model_dirs = glob.glob(pattern)
        if model_dirs:
            model_dirs.sort(reverse=True)  # Latest first
            all_models[algorithm] = model_dirs
        else:
            all_models[algorithm] = []
    
    return all_models


def run_comparative_analysis(algorithms, n_samples=500, n_background=50):
    """
    Run comparative analysis across multiple algorithms
    
    Args:
        algorithms (list): List of algorithms to compare
        n_samples (int): Number of samples for analysis
        n_background (int): Number of background samples
        
    Returns:
        dict: Comparative results
    """
    print("Running Comparative SHAP Analysis")
    print("=" * 80)
    
    all_results = {}
    
    for algorithm in algorithms:
        try:
            print(f"\nAnalyzing {algorithm.upper()}...")
            analyzer = UnifiedShapAnalyzer(algorithm)
            sample_data = analyzer.generate_sample_data(n_samples=n_samples)
            results = analyzer.run_shap_analysis(sample_data, n_background=n_background)
            all_results[algorithm] = results
            
        except Exception as e:
            print(f"Failed to analyze {algorithm.upper()}: {e}")
            continue
    
    if len(all_results) > 1:
        print(f"\nCreating comparative visualizations...")
        create_comparative_visualizations(all_results)
    
    return all_results


def create_comparative_visualizations(all_results):
    """Create comparative visualizations across algorithms"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir = f"comparative_analysis_{timestamp}"
    os.makedirs(comp_dir, exist_ok=True)
    
    print(f"Creating comparative visualizations in {comp_dir}/")
    
    # 1. Overall feature importance comparison
    plt.figure(figsize=(14, 10))
    
    algorithms = list(all_results.keys())
    feature_names = None
    importance_data = {}
    
    for algorithm, results in all_results.items():
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(results['shap_values'].values), axis=(0, 2))
        
        if feature_names is None:
            feature_names = [results['readable_feature_names'].get(f, f) for f in results['feature_names']]
        
        importance_data[results['config']['name']] = mean_abs_shap
    
    # Create comparison dataframe
    importance_df = pd.DataFrame(importance_data, index=feature_names)
    
    # Plot comparison
    ax = importance_df.plot(kind='barh', figsize=(12, 14), 
                           color=[all_results[alg]['config']['color'] for alg in algorithms])
    plt.title('Feature Importance Comparison Across Algorithms', fontsize=16, fontweight='bold')
    plt.xlabel('Mean |SHAP Value|')
    plt.ylabel('Patient Features')
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{comp_dir}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top 10 features heatmap
    plt.figure(figsize=(12, 8))
    
    # Get top 10 features overall
    overall_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    top_10_features = overall_importance.head(10).index
    
    heatmap_data = importance_df.loc[top_10_features]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Mean |SHAP Value|'})
    plt.title('Top 10 Features: Algorithm Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Algorithm')
    plt.ylabel('Patient Features')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{comp_dir}/top_features_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Treatment-specific comparison for key treatments
    key_treatments = ['AD Treatment_active', 'Antidepressant_active', 'Statin_active', 'No Medication_active']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, treatment in enumerate(key_treatments):
        if i >= len(axes):
            break
            
        treatment_data = {}
        
        for algorithm, results in all_results.items():
            if treatment in results['action_names']:
                action_idx = results['action_names'].index(treatment)
                treatment_importance = np.mean(np.abs(results['shap_values'].values[:, :, action_idx]), axis=0)
                treatment_data[results['config']['name']] = treatment_importance
        
        if treatment_data:
            treatment_df = pd.DataFrame(treatment_data, index=feature_names)
            top_features = treatment_df.mean(axis=1).sort_values(ascending=False).head(8)
            
            treatment_df.loc[top_features.index].plot(kind='barh', ax=axes[i],
                                                     color=[all_results[alg]['config']['color'] for alg in algorithms])
            
            readable_treatment = list(all_results.values())[0]['readable_action_names'].get(treatment, treatment)
            axes[i].set_title(f'{readable_treatment}', fontweight='bold')
            axes[i].set_xlabel('Mean |SHAP Value|')
            
            if i == 0:
                axes[i].legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                axes[i].legend().set_visible(False)
    
    plt.suptitle('Treatment-Specific Feature Importance by Algorithm', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{comp_dir}/treatment_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary statistics
    summary_stats = []
    for algorithm, results in all_results.items():
        stats = {
            'Algorithm': results['config']['name'],
            'Model Directory': os.path.basename(results['model_dir']),
            'Samples Analyzed': len(results['sample_data']),
            'Mean Feature Importance': np.mean(np.abs(results['shap_values'].values)),
            'Top Feature': feature_names[np.argmax(np.mean(np.abs(results['shap_values'].values), axis=(0, 2)))],
            'Top Feature Importance': np.max(np.mean(np.abs(results['shap_values'].values), axis=(0, 2)))
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(f"{comp_dir}/comparative_summary.csv", index=False)
    
    print(f"Comparative visualizations saved to {comp_dir}/")
    
    # Print summary
    print(f"\nCOMPARATIVE ANALYSIS SUMMARY")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(f"{row['Algorithm']}:")
        print(f"  Top Feature: {row['Top Feature']} ({row['Top Feature Importance']:.4f})")
        print(f"  Mean Importance: {row['Mean Feature Importance']:.4f}")
        print()
    
    return comp_dir


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified SHAP Analysis for RL Algorithms')
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'a2c', 'sac', 'all'], default='ppo',
                       help='Algorithm to analyze (default: ppo)')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Path to trained model directory (default: auto-find latest)')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of samples to generate for analysis (default: 500)')
    parser.add_argument('--n_background', type=int, default=50,
                       help='Number of background samples for SHAP explainer (default: 50)')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparative analysis across algorithms')
    
    args = parser.parse_args()
    
    print("Unified SHAP Analysis for Reinforcement Learning")
    print("=" * 80)
    
    try:
        if args.algorithm == 'all' or args.compare:
            # Find available algorithms
            all_models = find_all_models()
            available_algorithms = [alg for alg, models in all_models.items() if models]
            
            if not available_algorithms:
                print("No trained models found for any algorithm")
                return False
            
            print(f"Found models for: {', '.join([alg.upper() for alg in available_algorithms])}")
            
            if args.compare:
                comparative_results = run_comparative_analysis(available_algorithms, 
                                                             args.n_samples, args.n_background)
                print(f"\nComparative analysis completed!")
                return True
            else:
                # Run analysis for all available algorithms
                for algorithm in available_algorithms:
                    print(f"\n{'='*20} {algorithm.upper()} ANALYSIS {'='*20}")
                    analyzer = UnifiedShapAnalyzer(algorithm, args.model_dir)
                    sample_data = analyzer.generate_sample_data(n_samples=args.n_samples)
                    results = analyzer.run_shap_analysis(sample_data, n_background=args.n_background)
                    print(f"{algorithm.upper()} analysis completed!")
        else:
            # Single algorithm analysis
            analyzer = UnifiedShapAnalyzer(args.algorithm, args.model_dir)
            sample_data = analyzer.generate_sample_data(n_samples=args.n_samples)
            results = analyzer.run_shap_analysis(sample_data, n_background=args.n_background)
            print(f"\n{args.algorithm.upper()} SHAP Analysis completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"SHAP Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
