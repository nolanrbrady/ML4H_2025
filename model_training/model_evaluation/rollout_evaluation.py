import os
import sys
import json
import pandas as pd
import torch
import numpy as np
from typing import Tuple, Optional, List

"""
Transformer-only rollout evaluation using schema-defined input/output ordering.

This script autoregressively rolls out trajectories by:
- Reconstructing model inputs at each step as [true actions_t, predicted state_t, time_delta_t->t+1]
- Using a decoder-only transformer with a causal mask
- Post-processing outputs by rounding ordinal continuous variables and enforcing
  one-hot constraints within categorical groups (argmax), plus thresholding
  independent binary columns at 0.5.
"""

# Ensure model_training and package root ('autoreg') are importable
_model_training_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg_root_dir = os.path.abspath(os.path.join(_model_training_dir, '..'))
_current_dir = os.path.dirname(os.path.abspath(__file__))
for p in (_model_training_dir, _pkg_root_dir, _current_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

# Select model kind: 'transformer' (baseline) or 'moe' (Mixture-of-Experts)
MODEL_KIND = os.environ.get('MODEL_KIND', 'moe').lower()  # 'transformer' or 'moe'

if MODEL_KIND == 'moe':
    from train_moe_transformer import build_model_from_schema  # type: ignore
else:
    from train_transformer import build_model_from_schema  # type: ignore
from preprocessing import ORDINAL_VARS  # type: ignore
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import joblib  # For loading scalers
from sklearn.preprocessing import StandardScaler  # Type hints
from utils import (
    load_schema,
    derive_feature_sets_from_schema,
    print_feature_sets,
    load_validation_data,
    load_and_inspect_scalers,
    validate_scaler_coverage,
    inverse_scale_columns_inplace,
    validate_train_val_alignment,
    build_category_indices,
    autoregressive_rollout_for_subject,
)
try:
    from hyppo.ksample import MMD as HyppoMMD  # For MMD tests
except Exception:
    HyppoMMD = None  # Graceful fallback if hyppo is not installed
try:
    from skbio.stats.distance import mantel as skbio_mantel  # Mantel from scikit-bio
except Exception:
    skbio_mantel = None
from scipy.stats import chi2

# ------------------ Timepoint Correlation Utilities ------------------
def _vectorize_upper_tri(mat: np.ndarray) -> np.ndarray:
    """Return the vectorized upper triangle (k=1) of a square matrix."""
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]

def _safe_corrcoef_rows(X: np.ndarray) -> np.ndarray:
    """
    Compute a timepoint-by-timepoint Pearson correlation matrix between row vectors
    (correlation across features). Handles constant rows by returning 0 correlation
    with other rows and 1 on the diagonal.
    X shape: (T, F)
    """
    T, F = X.shape
    if F < 2 or T < 2:
        return np.eye(T, dtype=float)
    # Row-wise standardization across features
    X = X.astype(float)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, ddof=1, keepdims=True)
    # Avoid division by zero
    safe_stds = stds.copy()
    safe_stds[safe_stds == 0] = 1.0
    Z = (X - means) / safe_stds
    # Compute sample correlation across features; scale by (F-1)
    C = (Z @ Z.T) / max(F - 1, 1)
    # Set rows that were constant back to zero corr off-diagonal, one on diagonal
    const_mask = (stds.squeeze(-1) == 0)
    if np.any(const_mask):
        C[const_mask, :] = 0.0
        C[:, const_mask] = 0.0
        np.fill_diagonal(C, 1.0)
    # Numeric cleanup
    C = np.clip(np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    return C

def _fisher_combined_p(pvals: np.ndarray) -> float:
    pvals = pvals[np.isfinite(pvals) & (pvals > 0)]
    if pvals.size == 0:
        return np.nan
    stat = -2.0 * np.sum(np.log(pvals))
    df = 2 * pvals.size
    return float(chi2.sf(stat, df))

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(42)

# Define key column names
SUBJECT_ID_COL = 'subject_id'
TIME_COL = 'months_since_bl'  # Absolute time column
# Relative time-gap feature used as model input; utils will alias to 'next_visit_months' if present
TIME_GAP_COL = 'time_delta'

# Load the validation set and ensure unique chronological visits per subject
X_val_df, X_val_df_unique_visits = load_validation_data('../X_test.csv', SUBJECT_ID_COL, TIME_COL, TIME_GAP_COL)

SCHEMA_PATH = "../columns_schema.json"
schema = load_schema(SCHEMA_PATH)
model_input_cols, action_cols, observation_cols, state_cols = derive_feature_sets_from_schema(schema, TIME_GAP_COL)

# Determine the effective time-gap column name present in model_input_cols
EFFECTIVE_GAP_COL = (
    'next_visit_months' if 'next_visit_months' in model_input_cols else TIME_GAP_COL
)

# --- Print feature list orders for transparency (reflecting configuration) ---
print_feature_sets(model_input_cols, action_cols, observation_cols, state_cols)

# Define model input and target sizes based on the derived feature lists
model_input_size_derived = len(model_input_cols)
model_target_size_derived = len(observation_cols)

print(f"\n--- Derived Model Dimensions (based on configuration) ---")
print(f"Model input size: {model_input_size_derived}")
print(f"Model target size: {model_target_size_derived}")
print(f"--- End Derived Model Dimensions ---\n")

base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

#============
# Base Model
#============
preferred = os.path.join(base_dir, "best_moe_transformer_model.pt" if MODEL_KIND == 'moe' else "best_transformer_model.pt")

#============
# Fine-tuned model
#============
# preferred = os.path.join(base_dir, "best_moe_transformer_SFT.pt" if MODEL_KIND == 'moe' else "best_transformer_model.pt")

# Alternative type model
alt_moe_sft = os.path.join(base_dir, "best_moe_transfomer_model.pt")  # legacy typo compatibility
best_path = preferred if os.path.exists(preferred) else (alt_moe_sft if os.path.exists(alt_moe_sft) else preferred)
# Build transformer from schema; aux contains index mapping from schema to heads

# Initialize the model
if MODEL_KIND == 'transformer':
    # Model parameters
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.2

    model, aux = build_model_from_schema(SCHEMA_PATH, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
elif MODEL_KIND == 'moe':
    # Model parameters
    d_model = 256
    nhead = 4
    num_layers = 3
    dim_feedforward = 768
    dropout = 0.3
    num_experts = 8
    num_experts_per_tok = 1

    model, aux = build_model_from_schema(
        SCHEMA_PATH,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

cont_idx = aux.get("cont_idx", [])    # indices of continuous/ordinal outputs within observation_cols
bin_idx = aux.get("bin_idx", [])      # indices of binary outputs within observation_cols
if os.path.exists(best_path):
    ckpt = torch.load(best_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    print(f"Loaded Transformer from '{best_path}'")
else:
    print(f"Warning: best checkpoint not found at {best_path}. Using randomly-initialized weights.")
model = model.to(device)
model.eval()

# Build categorical group indices for argmax enforcement and a set of independent binary cols.
# The schema-defined y_categorical_groups maps a group name (e.g., 'PTMARRY')
# to its one-hot columns (e.g., ['PTMARRY_Divorced', ...]). We enforce a valid
# one-hot by argmax within each group, and threshold the remaining binary columns at 0.5.
group_indices, group_member_indices = build_category_indices(schema, observation_cols)
independent_bin_indices = [i for i in bin_idx if i not in set(group_member_indices)]

# Load the scalers
scaler_X, scaler_y = load_and_inspect_scalers('../scaler_X.joblib', '../scaler_y.joblib')

def _inverse_scale_columns_inplace(df: pd.DataFrame, cols: list, scaler, scaler_name: str):
    """Backwards-compatible wrapper around utils.inverse_scale_columns_inplace."""
    inverse_scale_columns_inplace(df, cols, scaler, scaler_name)

# --- Validate Scaler Feature Names and Coverage ---
validate_scaler_coverage(
    scaler_X,
    scaler_y,
    trajectory_cols=list(set(state_cols) | {TIME_COL, EFFECTIVE_GAP_COL}),
    observation_cols=observation_cols,
)

features_to_plot_requested = ['WholeBrain', 'ADNI_MEM']
PLOTTING_SUBJECT_LIMIT = 5 # Define a constant for how many subjects to plot

subjects = X_val_df[SUBJECT_ID_COL].unique()
num_subjects_plotted = 0 # Counter for subjects for whom plots have been generated

# Lists to store all trajectories for MMD analysis
all_true_trajectories_targets = []
all_predicted_trajectories_targets = []
subject_mantel_results: List[dict] = []
subject_mantel_perm_rs: List[np.ndarray] = []

# --- Critical Validation: Compare column orders between train and validation sets ---
print("Validating column order consistency between training and validation datasets...")
y_train_effective = validate_train_val_alignment(
    '../X_train.csv',
    '../y_train.csv',
    observation_cols,
    SUBJECT_ID_COL,
    model_input_cols,
)

for subject_idx, subject in enumerate(subjects):
    print(f"Processing subject: {subject} ({subject_idx + 1}/{len(subjects)})")
    # Use X_val_df_unique_visits to ensure a unique, chronological sequence of visits
    subject_true_data_df = X_val_df_unique_visits[X_val_df_unique_visits[SUBJECT_ID_COL] == subject].sort_values(by=TIME_COL).reset_index(drop=True)

    if len(subject_true_data_df) < 6:
        print(f"Subject {subject} has fewer than 2 visits. Skipping.")
        continue

    predicted_states_list_of_dicts = autoregressive_rollout_for_subject(
        subject_true_df=subject_true_data_df,
        model=model,
        device=device,
        model_input_cols=model_input_cols,
        observation_cols=observation_cols,
        state_cols=state_cols,
        action_cols=action_cols,
        time_col=TIME_COL,
        time_delta_col=EFFECTIVE_GAP_COL,
        cont_idx=cont_idx,
        bin_idx=bin_idx,
        group_indices=group_indices,
        independent_bin_indices=independent_bin_indices,
        ordinal_vars=ORDINAL_VARS,
        schema=schema,
    )

    predicted_trajectory_df = pd.DataFrame(predicted_states_list_of_dicts)
    # Ensure columns in predicted_trajectory_df are state_cols + [TIME_COL]
    # This might involve reordering if dictionary iteration order wasn't guaranteed.
    # Use dict.fromkeys to preserve order of FCT and ensure TIME_COL is present once.
    final_ordered_cols_for_df = list(dict.fromkeys(state_cols + [TIME_COL]))
    predicted_trajectory_df = predicted_trajectory_df[final_ordered_cols_for_df]

    # Add effective time-gap column from the true trajectory to predicted_trajectory_df.
    # This is necessary if scaler_X was trained on the gap feature; it will be needed for unscaling.
    if EFFECTIVE_GAP_COL in subject_true_data_df.columns:
        gap_vals = subject_true_data_df[EFFECTIVE_GAP_COL].values
    elif TIME_GAP_COL in subject_true_data_df.columns:
        gap_vals = subject_true_data_df[TIME_GAP_COL].values
    else:
        # Compute gap from consecutive times as a fallback
        gap_vals = np.diff(subject_true_data_df[TIME_COL].to_numpy(), prepend=subject_true_data_df[TIME_COL].iloc[0])
    predicted_trajectory_df[EFFECTIVE_GAP_COL] = gap_vals
    # If the scaler expects the alternate name, add it too for coverage
    if scaler_X and hasattr(scaler_X, 'feature_names_in_'):
        scaler_feats = set(list(scaler_X.feature_names_in_))
        alt_name = 'time_delta' if EFFECTIVE_GAP_COL == 'next_visit_months' else 'next_visit_months'
        if alt_name in scaler_feats:
            predicted_trajectory_df[alt_name] = gap_vals

    # --- Unscale data for plotting if scalers are loaded ---
    unscaled_subject_true_data_df = subject_true_data_df.copy()
    unscaled_predicted_trajectory_df = predicted_trajectory_df.copy()

    if scaler_X:
        print("--- Unscaling True Trajectory (using scaler_X, per-column) ---")
        # Use column-wise inverse scaling to avoid requiring a full feature matrix alignment
        if hasattr(scaler_X, 'feature_names_in_'):
            scaler_X_known_features = list(scaler_X.feature_names_in_)
            true_cols_to_unscale = [col for col in scaler_X_known_features if col in unscaled_subject_true_data_df.columns]
            if true_cols_to_unscale:
                _inverse_scale_columns_inplace(unscaled_subject_true_data_df, true_cols_to_unscale, scaler_X, 'scaler_X')
                print(f"Unscaled {len(true_cols_to_unscale)} columns in true trajectory using scaler_X (per-column).")
        else:
            print("Warning: scaler_X loaded but 'feature_names_in_' attribute is missing. Cannot reliably unscale true trajectory.")

    if scaler_y and scaler_X:
        print("\n--- Unscaling Predicted Trajectory (scaler_y for targets, scaler_X for actions; per-column) ---")
        # 1. Unscale target features with scaler_y
        if hasattr(scaler_y, 'feature_names_in_'):
            scaler_y_known_features = list(scaler_y.feature_names_in_)
            pred_y_cols = [col for col in observation_cols if col in scaler_y_known_features and col in unscaled_predicted_trajectory_df.columns]
            if pred_y_cols:
                _inverse_scale_columns_inplace(unscaled_predicted_trajectory_df, pred_y_cols, scaler_y, 'scaler_y')
                print(f"Unscaled {len(pred_y_cols)} target columns in predicted trajectory using scaler_y (per-column).")
        else:
            print("Warning: scaler_y loaded but 'feature_names_in_' attribute is missing. Cannot reliably unscale predicted trajectory targets.")

        # 1b. Fallback: Unscale any remaining target features with scaler_X if available
        if hasattr(scaler_X, 'feature_names_in_'):
            scaler_X_known_features = set(scaler_X.feature_names_in_)
            remaining_pred_y_cols = [
                col for col in observation_cols
                if col not in (list(scaler_y.feature_names_in_) if hasattr(scaler_y, 'feature_names_in_') else [])
                and col in scaler_X_known_features
                and col in unscaled_predicted_trajectory_df.columns
            ]
            if remaining_pred_y_cols:
                _inverse_scale_columns_inplace(unscaled_predicted_trajectory_df, remaining_pred_y_cols, scaler_X, 'scaler_X (fallback for targets)')
                print(f"Unscaled {len(remaining_pred_y_cols)} additional target columns using scaler_X as fallback.")

        # 2. Unscale action features with scaler_X
        if hasattr(scaler_X, 'feature_names_in_'):
            scaler_X_known_features = list(scaler_X.feature_names_in_)
            pred_x_cols = [col for col in action_cols if col in scaler_X_known_features and col in unscaled_predicted_trajectory_df.columns]
            if pred_x_cols:
                _inverse_scale_columns_inplace(unscaled_predicted_trajectory_df, pred_x_cols, scaler_X, 'scaler_X')
                print(f"Unscaled {len(pred_x_cols)} action columns in predicted trajectory using scaler_X (per-column).")
        else:
            print("Warning: scaler_X loaded but 'feature_names_in_' attribute is missing. Cannot reliably unscale predicted trajectory actions.")

    if not scaler_X or not scaler_y:
        print("Scalers not loaded. Plotting with original (potentially scaled) values.")
    # --- End Unscaling ---

    # Accumulate data for MMD analysis using unscaled data
    # We are interested in the distribution of observation_cols
    if not unscaled_subject_true_data_df.empty and not unscaled_predicted_trajectory_df.empty:
        true_targets_for_subject = unscaled_subject_true_data_df[observation_cols].values
        pred_targets_for_subject = unscaled_predicted_trajectory_df[observation_cols].values
        
        # Ensure both have the same number of timepoints (should be true by construction)
        if true_targets_for_subject.shape[0] == pred_targets_for_subject.shape[0] and true_targets_for_subject.shape[0] > 0:
            all_true_trajectories_targets.append(true_targets_for_subject)
            all_predicted_trajectories_targets.append(pred_targets_for_subject)
            # --- Mantel test per subject on pairwise correlation structure across timepoints ---
            try:
                T, F = true_targets_for_subject.shape
                if T >= 3 and F >= 2:
                    C_true = _safe_corrcoef_rows(true_targets_for_subject)
                    C_pred = _safe_corrcoef_rows(pred_targets_for_subject)
                    # Convert similarities to distances for Mantel
                    D_true = 1.0 - C_true
                    D_pred = 1.0 - C_pred
                    np.fill_diagonal(D_true, 0.0)
                    np.fill_diagonal(D_pred, 0.0)
                    if skbio_mantel is None:
                        raise ImportError("scikit-bio is required for Mantel test. Please install scikit-bio.")
                    r_mantel, p_mantel, _ = skbio_mantel(
                        D_true, D_pred, method='pearson', permutations=999, alternative='two-sided'
                    )
                    subject_mantel_results.append({
                        'subject_id': subject,
                        'timepoints': int(T),
                        'features': int(F),
                        'mantel_r': float(r_mantel) if r_mantel is not None else np.nan,
                        'mantel_p': float(p_mantel) if p_mantel is not None else np.nan,
                    })
                else:
                    subject_mantel_results.append({
                        'subject_id': subject,
                        'timepoints': int(true_targets_for_subject.shape[0]),
                        'features': int(true_targets_for_subject.shape[1]),
                        'mantel_r': np.nan,
                        'mantel_p': np.nan,
                    })
            except Exception as e:
                print(f"Warning: Mantel test failed for subject {subject}: {e}")
                subject_mantel_results.append({
                    'subject_id': subject,
                    'timepoints': int(true_targets_for_subject.shape[0]),
                    'features': int(true_targets_for_subject.shape[1]),
                    'mantel_r': np.nan,
                    'mantel_p': np.nan,
                })
        else:
            print(f"Warning: Skipping subject {subject} for MMD due to mismatched or empty trajectory lengths after unscaling.")
            print(f"  True shape: {true_targets_for_subject.shape}, Predicted shape: {pred_targets_for_subject.shape}")

    # Plotting (limited by num_subjects_plotted)
    if num_subjects_plotted < PLOTTING_SUBJECT_LIMIT:
        available_features_for_plot = [f for f in features_to_plot_requested if f in state_cols]
        if TIME_COL not in available_features_for_plot and TIME_COL in predicted_trajectory_df.columns:
             if TIME_COL in features_to_plot_requested or True: 
                available_features_for_plot.append(TIME_COL)

        if not available_features_for_plot:
            print(f"No requested or available features for plotting for subject {subject}. Skipping plot generation for this subject.")
        else:
            num_features = len(available_features_for_plot)
            # Create a figure with 2 plots per feature (unscaled and scaled)
            plt.figure(figsize=(14, 6 * num_features))
            plt.suptitle(f"Subject {subject} - Trajectory Comparison", fontsize=16)
            
            time_indices = range(len(subject_true_data_df))

            # Debug: show a quick summary for the first subject to verify scaling differences
            if num_subjects_plotted == 0 and available_features_for_plot:
                try:
                    for sample_feat in available_features_for_plot:
                        tvu = unscaled_subject_true_data_df[sample_feat]
                        pvu = unscaled_predicted_trajectory_df[sample_feat]
                        tvs = subject_true_data_df[sample_feat]
                        pvs = predicted_trajectory_df[sample_feat]
                        print(f"Scaling check [{sample_feat}]: true_scaled[min,max]=({tvs.min():.4g},{tvs.max():.4g}), true_unscaled[min,max]=({tvu.min():.4g},{tvu.max():.4g}); pred_scaled[min,max]=({pvs.min():.4g},{pvs.max():.4g}), pred_unscaled[min,max]=({pvu.min():.4g},{pvu.max():.4g})")
                        # Equality checks
                        try:
                            eq_true = np.allclose(tvs.to_numpy(dtype=float), tvu.to_numpy(dtype=float), atol=1e-6, rtol=1e-6)
                            eq_pred = np.allclose(pvs.to_numpy(dtype=float), pvu.to_numpy(dtype=float), atol=1e-6, rtol=1e-6)
                            print(f"  identical true scaled/unscaled: {eq_true}; identical pred scaled/unscaled: {eq_pred}")
                        except Exception:
                            pass
                        # Also print scaler stats if available
                        if scaler_X and hasattr(scaler_X, 'feature_names_in_') and sample_feat in list(scaler_X.feature_names_in_):
                            j = list(scaler_X.feature_names_in_).index(sample_feat)
                            print(f"  scaler_X: mean={getattr(scaler_X, 'mean_', [None]*999)[j] if hasattr(scaler_X,'mean_') else 'NA'}, scale={getattr(scaler_X, 'scale_', [None]*999)[j] if hasattr(scaler_X,'scale_') else 'NA'}")
                        if scaler_y and hasattr(scaler_y, 'feature_names_in_') and sample_feat in list(scaler_y.feature_names_in_):
                            j = list(scaler_y.feature_names_in_).index(sample_feat)
                            print(f"  scaler_Y: mean={getattr(scaler_y, 'mean_', [None]*999)[j] if hasattr(scaler_y,'mean_') else 'NA'}, scale={getattr(scaler_y, 'scale_', [None]*999)[j] if hasattr(scaler_y,'scale_') else 'NA'}")
                except Exception as e:
                    print(f"Scaling debug failed: {e}")

            for i, feature_name in enumerate(available_features_for_plot):
                # --- Plot 1: Unscaled Data ---
                plt.subplot(num_features, 2, 2 * i + 1)
                
                true_values_unscaled = unscaled_subject_true_data_df[feature_name]
                predicted_values_unscaled = unscaled_predicted_trajectory_df[feature_name]
                # Detect if unscaled equals scaled (no-op due to missing scaler)
                try:
                    _eq_true = np.allclose(true_values_unscaled.to_numpy(dtype=float), subject_true_data_df[feature_name].to_numpy(dtype=float), atol=1e-6, rtol=1e-6)
                    _eq_pred = np.allclose(predicted_values_unscaled.to_numpy(dtype=float), predicted_trajectory_df[feature_name].to_numpy(dtype=float), atol=1e-6, rtol=1e-6)
                except Exception:
                    _eq_true = _eq_pred = False
                
                plt.plot(time_indices, true_values_unscaled, label=f"True Trajectory{' [same]' if _eq_true else ''}", marker='o', linestyle='-')
                plt.plot(time_indices, predicted_values_unscaled, label=f"Predicted Trajectory{' [same]' if _eq_pred else ''}", marker='x', linestyle='--')
                
                plt.title(f"{feature_name} (Original Units)")
                plt.xlabel("Visit Number (Chronological)")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)

                # --- Plot 2: Scaled Data ---
                plt.subplot(num_features, 2, 2 * i + 2)
                
                # Use the original (scaled) dataframes before the unscaling section
                true_values_scaled = subject_true_data_df[feature_name]
                predicted_values_scaled = predicted_trajectory_df[feature_name]
                
                plt.plot(time_indices, true_values_scaled, label='True Trajectory (Scaled)', marker='o', linestyle='-')
                plt.plot(time_indices, predicted_values_scaled, label='Predicted Trajectory (Scaled)', marker='x', linestyle='--')
                
                plt.title(f"{feature_name} (Standardized)")
                plt.xlabel("Visit Number (Chronological)")
                plt.ylabel("z-score")
                plt.legend()
                plt.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            num_subjects_plotted += 1 # Increment only if a plot was generated
    
# After the loop, print a summary of plotting.
print(f"\nFinished processing all {len(subjects)} subjects.")
if num_subjects_plotted > 0:
    print(f"Plotting was shown for the first {num_subjects_plotted} subjects with available data.")
elif PLOTTING_SUBJECT_LIMIT > 0:
    print(f"No subjects were plotted (limit was {PLOTTING_SUBJECT_LIMIT}, but no suitable data or all subjects processed after limit was reached).")
else:
    print("Plotting was disabled (PLOTTING_SUBJECT_LIMIT = 0).")

# --- MMD Analysis Section (Per-Feature using Hyppo) ---
print("\n--- Per-Feature Maximum Mean Discrepancy (MMD) Analysis (using Hyppo) ---")
if HyppoMMD is None:
    print("Hyppo is not available in this environment; skipping per-feature MMD analysis.")
elif not all_true_trajectories_targets or not all_predicted_trajectories_targets:
    print("No data collected for MMD analysis. Skipping.")
else:
    all_true_states_np = np.concatenate(all_true_trajectories_targets, axis=0)
    all_pred_states_np = np.concatenate(all_predicted_trajectories_targets, axis=0)

    if all_true_states_np.shape[0] < 2 or all_pred_states_np.shape[0] < 2: # Hyppo generally needs at least 2 samples in each set
        print(f"Insufficient samples for MMD analysis (True: {all_true_states_np.shape[0]}, Pred: {all_pred_states_np.shape[0]}). Skipping MMD.")
    elif all_true_states_np.shape[1] != len(observation_cols) or \
         all_pred_states_np.shape[1] != len(observation_cols):
        print(f"Mismatch in feature dimensions for MMD analysis. Expected {len(observation_cols)} features.")
        print(f"  True states shape: {all_true_states_np.shape}, Predicted states shape: {all_pred_states_np.shape}")
    else:
        print(f"Performing per-feature MMD analysis on {all_true_states_np.shape[0]} true state points and {all_pred_states_np.shape[0]} predicted state points.")
        print(f"Number of features for MMD: {len(observation_cols)}")
        
        mmd_results_per_feature = {}
        num_features_for_correction = len(observation_cols)

        for i, feature_name in enumerate(observation_cols):
            print(f"  Calculating MMD for feature: {feature_name}...")
            true_feature_data = all_true_states_np[:, i]
            pred_feature_data = all_pred_states_np[:, i]

            # Ensure data is 2D for hyppo and has enough samples
            true_feature_data_2d = true_feature_data.reshape(-1, 1)
            pred_feature_data_2d = pred_feature_data.reshape(-1, 1)

            if true_feature_data_2d.shape[0] < 2 or pred_feature_data_2d.shape[0] < 2:
                print(f"    Skipping MMD for {feature_name} due to insufficient samples for hyppo (True: {true_feature_data_2d.shape[0]}, Pred: {pred_feature_data_2d.shape[0]}).")
                mmd_results_per_feature[feature_name] = {'mmd_stat': np.nan, 'p_value': np.nan, 'p_value_corrected': np.nan}
                continue
            
            try:
                # For 1D feature, explicitly set gamma for RBF kernel.
                # Note: Hyppo uses an unbiased MMD estimator that can return negative values
                # when distributions are very similar. This is normal behavior - focus on p-values.
                mmd_test_univar = HyppoMMD(compute_kernel="rbf")
                stat, p_val = mmd_test_univar.test(true_feature_data_2d, pred_feature_data_2d, reps=5000, workers=-1, random_state=42)
                
                p_val_corrected = min(p_val * num_features_for_correction, 1.0) # Bonferroni correction
                mmd_results_per_feature[feature_name] = {'mmd_stat': stat, 'p_value': p_val, 'p_value_corrected': p_val_corrected}
                print(f"    {feature_name}: MMD Stat = {stat:.4e}, P-value = {p_val:.4f}, P-value (Bonf.) = {p_val_corrected:.4f}")
            except Exception as e:
                print(f"    Hyppo MMD calculation failed for feature {feature_name}: {e}")
                mmd_results_per_feature[feature_name] = {'mmd_stat': np.nan, 'p_value': np.nan, 'p_value_corrected': np.nan}

        print("\nPer-Feature MMD Analysis Summary:")
        significantly_different_uncorrected = 0
        significantly_different_corrected = 0
        alpha = 0.05

        for feature_name, results in mmd_results_per_feature.items():
            if not np.isnan(results['p_value']) and results['p_value'] < alpha:
                significantly_different_uncorrected += 1
            if not np.isnan(results['p_value_corrected']) and results['p_value_corrected'] < alpha:
                significantly_different_corrected += 1

        print(f"Total features analyzed: {len(mmd_results_per_feature)}")
        num_valid_results = sum(1 for res in mmd_results_per_feature.values() if not np.isnan(res['mmd_stat']))
        print(f"Number of features with valid MMD results: {num_valid_results}")
        if num_valid_results > 0:
            print(f"Number of features significantly different (p < {alpha}, uncorrected): {significantly_different_uncorrected}")
            print(f"Number of features significantly different (p < {alpha}, Bonferroni corrected): {significantly_different_corrected}")
        else:
            print("No valid MMD results to summarize significance for per-feature tests.")

    # --- Multivariate MMD Analysis (using Hyppo) ---
print("\n--- Multivariate MMD Analysis (all target features, using Hyppo) ---")
if HyppoMMD is None:
    print("Hyppo is not available in this environment; skipping multivariate MMD analysis.")
elif all_true_states_np.shape[0] < 2 or all_pred_states_np.shape[0] < 2:
    print(f"Insufficient samples for multivariate MMD test (True: {all_true_states_np.shape[0]}, Pred: {all_pred_states_np.shape[0]}). Skipping.")
elif all_true_states_np.shape[1] == 0: # No features
    print("No features available for multivariate MMD test. Skipping.")
else:
        try:
            # For multivariate, let hyppo use its default RBF kernel (often with median heuristic for gamma)
            # Using 1000 permutations for the overall test to avoid the warning.
            multivar_mmd_test = HyppoMMD(compute_kernel="rbf") 
            
            print(f"Performing multivariate MMD on all {all_true_states_np.shape[1]} target features simultaneously.")
            print(f"  True data shape: {all_true_states_np.shape}, Predicted data shape: {all_pred_states_np.shape}")
            
            # Check for NaNs or Infs which might cause issues with hyppo
            if np.any(np.isnan(all_true_states_np)) or np.any(np.isinf(all_true_states_np)) or \
               np.any(np.isnan(all_pred_states_np)) or np.any(np.isinf(all_pred_states_np)):
                print("Warning: NaNs or Infs detected in data for multivariate MMD. This may lead to errors or unreliable results.")

            mv_stat, mv_p_value = multivar_mmd_test.test(all_true_states_np, all_pred_states_np, reps=1000, workers=-1, random_state=42)
            
            print(f"\nOverall Multivariate MMD Test Results (all {all_true_states_np.shape[1]} target features):")
            print(f"  MMD Statistic: {mv_stat:.4e}")
            print(f"  P-value: {mv_p_value:.4f}")
            if mv_p_value < alpha: # Using the same alpha as per-feature tests
                print(f"  The overall distributions of true and predicted trajectories (all target features considered together) are significantly different (p < {alpha}).")
            else:
                print(f"  The overall distributions of true and predicted trajectories (all target features considered together) are NOT significantly different (p >= {alpha}).")
                
        except Exception as e:
            print(f"Multivariate Hyppo MMD calculation failed: {e}")

# --- Mantel Test Group Summary ---
print("\n--- Mantel Test (pairwise timepoint correlation structure) ---")
if not subject_mantel_results:
    print("No per-subject Mantel results collected.")
else:
    df_mantel = pd.DataFrame(subject_mantel_results)
    valid = df_mantel.dropna(subset=['mantel_r', 'mantel_p'])
    print(f"Subjects evaluated: {len(df_mantel)}; valid Mantel: {len(valid)}")
    if not valid.empty:
        mean_r = valid['mantel_r'].mean()
        median_r = valid['mantel_r'].median()
        # Benjamini-Hochberg FDR
        pvals = valid['mantel_p'].to_numpy()
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        thresh = (np.arange(1, m + 1) / m) * 0.05
        passed = ranked <= thresh
        num_sig = int(passed.sum()) if np.any(passed) else 0
        print(f"Mean Mantel r: {mean_r:.4f}")
        print(f"Median Mantel r: {median_r:.4f}")
        print(f"Significant (BH-FDR 0.05): {num_sig}/{m}")
        # Group-level combined p-value via Fisher's method (uses per-subject permutation p-values)
        try:
            group_p_fisher = _fisher_combined_p(valid['mantel_p'].to_numpy())
            if np.isfinite(group_p_fisher):
                print(f"Group-level combined p-value (Fisher): {group_p_fisher:.4e}")
        except Exception as e:
            print(f"Warning computing group-level Fisher p-value: {e}")
        # Optional: show a few examples
        top_examples = valid.sort_values('mantel_r', ascending=False).head(3)
        print("Top subjects by Mantel r (id, r, p):")
        for _, row in top_examples.iterrows():
            print(f"  {row['subject_id']}: r={row['mantel_r']:.4f}, p={row['mantel_p']:.4f}, T={int(row['timepoints'])}, F={int(row['features'])}")
        # Save per-subject results with BH-FDR adjusted p-values
        df_out = df_mantel.copy()
        # Initialize adjusted p as NaN
        df_out['mantel_p_fdr_bh'] = np.nan
        if m > 0:
            # Compute BH for valid entries and map back
            bh_adj = np.empty(m, dtype=float)
            bh_adj[:] = np.nan
            # BH adjusted p-values (monotone increasing)
            ranked_adj = ranked * (m / np.arange(1, m + 1))
            ranked_adj = np.minimum.accumulate(ranked_adj[::-1])[::-1]
            # Place back to original order
            bh_adj[order] = np.clip(ranked_adj, 0.0, 1.0)
            df_out.loc[valid.index, 'mantel_p_fdr_bh'] = bh_adj
            df_out['significant_fdr_0.05'] = df_out['mantel_p_fdr_bh'].le(0.05)
        out_path = Path(__file__).resolve().parent / 'mantel_results_rollout.csv'
        try:
            df_out.to_csv(out_path, index=False)
            print(f"Saved Mantel per-subject results to {out_path}")
        except Exception as e:
            print(f"Warning: failed to save Mantel results CSV: {e}")
    else:
        print("No valid Mantel correlations to summarize.")

print("\nScript execution finished.")
