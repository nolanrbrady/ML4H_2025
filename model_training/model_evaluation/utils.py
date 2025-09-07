"""Utilities for model evaluation.

Each function has a single responsibility and uses clear names
to improve readability and maintainability of the evaluation scripts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib


# --------------------------- Schema / Features ---------------------------

def load_schema(schema_path: str | Path) -> Dict:
    """Load the JSON schema that defines feature ordering.

    Returns the parsed dictionary.
    """
    with open(schema_path, "r") as f:
        return json.load(f)


def derive_feature_sets_from_schema(
    schema: Dict,
    time_delta_col: str,
    action_suffix: str = "_active",
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return (model_input_cols, action_cols, observation_cols, state_cols) from schema.

    - model_input_cols: ordering used to build model inputs
    - action_cols: action features (from schema if available; else suffix heuristic)
    - observation_cols: target features predicted by the model
    - state_cols: model_input_cols without the time delta feature
    """
    model_input_cols = list(schema.get("model_input_cols", []))
    if not model_input_cols:
        raise ValueError("Schema missing 'model_input_cols'")

    observation_cols = list(schema.get("observation_cols", []))
    if not observation_cols:
        raise ValueError("Schema missing 'observation_cols'")

    action_cols = list(schema.get("action_cols", [])) or [
        col for col in model_input_cols if col.endswith(action_suffix)
    ]
    # Determine which time-gap column to exclude from state_cols.
    # Prefer the provided time_delta_col if present; otherwise fall back to a known alias.
    gap_alias = None
    if time_delta_col in model_input_cols:
        gap_alias = time_delta_col
    elif "next_visit_months" in model_input_cols:
        gap_alias = "next_visit_months"
    elif "time_delta" in model_input_cols:
        gap_alias = "time_delta"
    state_cols = [c for c in model_input_cols if c != gap_alias]
    return model_input_cols, action_cols, observation_cols, state_cols


def print_feature_sets(model_input_cols: Sequence[str],
                       action_cols: Sequence[str],
                       observation_cols: Sequence[str],
                       state_cols: Sequence[str]) -> None:
    """Log configured feature sets for transparency."""
    print("\n--- Feature Set Definitions and Order (Reflecting Configuration) ---")
    print(f"1. model_input_cols (order for model tensor, {len(model_input_cols)}):")
    print(f"   {list(model_input_cols)}")
    print(f"2. action_cols ({len(action_cols)}):")
    print(f"   {list(action_cols)}")
    print(f"3. observation_cols (predicted by model, {len(observation_cols)}):")
    print(f"   {list(observation_cols)}")
    print(f"4. state_cols (model_input_cols without time-gap feature, {len(state_cols)}):")
    print(f"   {list(state_cols)}")
    combined = list(action_cols) + list(observation_cols)
    if list(state_cols) == combined:
        print("   INFO: state_cols is ordered as action_cols followed by observation_cols.")
    else:
        print("   INFO: state_cols is NOT strictly ordered as action_cols then observation_cols.")
    print("--- End of Feature Set Definitions ---\n")


# --------------------------- Data Loading / Validation ---------------------------

def load_validation_data(x_val_path: str | Path,
                         subject_id_col: str,
                         time_col: str,
                         time_delta_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load X_val.csv and return (full_df, unique_visits_df) sorted per subject by time.

    Validates presence of required time columns.
    """
    X_val_df = pd.read_csv(x_val_path)

    if time_col not in X_val_df.columns:
        raise ValueError(
            f"Critical Error: The expected time column '{time_col}' was not found in {x_val_path}. "
            f"Available columns: {list(X_val_df.columns)}"
        )
    # Ensure a time-gap column exists for downstream utilities.
    if time_delta_col not in X_val_df.columns:
        # If new preprocessing exposed 'next_visit_months', mirror it into time_delta_col for compatibility.
        if "next_visit_months" in X_val_df.columns:
            print(f"INFO: '{time_delta_col}' missing in X_val; using 'next_visit_months' as a substitute.")
            X_val_df[time_delta_col] = X_val_df["next_visit_months"].astype(float)
        else:
            # As a final fallback, compute forward differences per subject
            print(f"INFO: '{time_delta_col}' missing in X_val; computing it from consecutive '{time_col}' values per subject.")
            X_val_df = X_val_df.sort_values([subject_id_col, time_col])
            forward = X_val_df.groupby(subject_id_col)[time_col].shift(-1)
            X_val_df[time_delta_col] = (forward - X_val_df[time_col]).fillna(0.0)

    unique_visits = X_val_df.drop_duplicates(subset=[subject_id_col, time_col], keep="first").copy()
    return X_val_df, unique_visits


def validate_train_val_alignment(x_train_path: str | Path,
                                 y_train_path: str | Path,
                                 schema_observation_cols: Sequence[str],
                                 subject_id_col: str,
                                 model_input_cols: Sequence[str]) -> Optional[List[str]]:
    """Validate train/val alignment and return y_train effective target list.

    - Compares X_train (excluding subject_id) to model_input_cols and logs differences.
    - Reads y_train columns, filters out non-target extras (e.g., IDs), and returns the list of
      y_train columns restricted to target features, preserving order found in y_train.
    - Returns None if files are missing.
    """
    try:
        X_train_df = pd.read_csv(x_train_path)
        y_train_df = pd.read_csv(y_train_path)
    except FileNotFoundError as e:
        print(f"Warning: Could not load training datasets for validation: {e}")
        print("Proceeding without column order validation - ensure preprocessing consistency manually.")
        return None

    # Check X_train order vs. schema-defined model_input_cols (ignoring id)
    X_train_cols = list(X_train_df.columns)
    X_train_cols_wo_id = [c for c in X_train_cols if c != subject_id_col]
    if X_train_cols_wo_id != list(model_input_cols):
        print("WARNING: Column order/content mismatch between X_train.csv and schema model_input_cols (ignoring subject_id).")
        print(f"X_train (without '{subject_id_col}') columns ({len(X_train_cols_wo_id)}): {X_train_cols_wo_id}")
        print(f"Schema model_input_cols ({len(model_input_cols)}): {list(model_input_cols)}")
        set_train = set(X_train_cols_wo_id)
        set_schema = set(model_input_cols)
        missing_in_schema = set_train - set_schema
        extra_in_schema = set_schema - set_train
        if missing_in_schema:
            print(f"Columns present in X_train (excluding id) but missing in schema model_input_cols: {sorted(missing_in_schema)}")
        if extra_in_schema:
            print(f"Columns present in schema model_input_cols but missing in X_train (excluding id): {sorted(extra_in_schema)}")
        print("Proceeding with schema-defined ordering for inference.")
    print("✓ Column order validation passed for X datasets:")
    print(f"  X datasets: {len(X_train_cols)} columns in consistent order with derived model inputs")

    # Prepare y_train effective columns (filter out extras not in observation targets)
    y_train_cols = list(y_train_df.columns)
    extras_in_y_train = [c for c in y_train_cols if c not in schema_observation_cols]
    if extras_in_y_train:
        print("INFO: y_train.csv contains non-target columns that will be ignored for validation and alignment:")
        print(f"      Extras in y_train.csv not in model targets: {extras_in_y_train}")
    y_train_effective = [c for c in y_train_cols if c in schema_observation_cols]

    # Validate content match after removing extras
    if set(schema_observation_cols) != set(y_train_effective):
        print("ERROR: Content mismatch between derived model target features and y_train.csv columns (after removing non-target extras)!")
        print(f"Derived Model target features ({len(schema_observation_cols)} from schema): {list(schema_observation_cols)}")
        print(f"y_train effective columns ({len(y_train_effective)} from y_train.csv): {y_train_effective}")
        missing_in_y_train = sorted(list(set(schema_observation_cols) - set(y_train_effective)))
        extra_in_y_effective = sorted(list(set(y_train_effective) - set(schema_observation_cols)))
        if missing_in_y_train:
            print(f"  Target features missing from y_train.csv: {missing_in_y_train}")
        if extra_in_y_effective:
            print(f"  Unexpected target-like features present only in y_train.csv: {extra_in_y_effective}")
        raise ValueError("Content mismatch for model target features vs y_train.csv (post-filter). Check feature extraction logic and schema alignment.")

    return y_train_effective


# --------------------------- Scalers ---------------------------

def load_and_inspect_scalers(scaler_X_path: str | Path,
                             scaler_y_path: str | Path) -> Tuple[Optional[object], Optional[object]]:
    """Load scalers and print a concise inspection of their features."""
    try:
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        print(f"Scalers loaded from {scaler_X_path} and {scaler_y_path}")
    except FileNotFoundError:
        print(f"Error: Scaler files not found at {scaler_X_path} or {scaler_y_path}. Plotting will use scaled values.")
        return None, None
    except Exception as e:
        print(f"Error loading scalers: {e}. Plotting will use scaled values.")
        return None, None

    print("\n--- Inspecting Loaded Scalers ---")
    if scaler_X:
        print("Inspecting scaler_X:")
        try:
            if hasattr(scaler_X, "feature_names_in_"):
                print(f"  scaler_X.feature_names_in_ ({len(scaler_X.feature_names_in_)} features):")
                print(f"    {list(scaler_X.feature_names_in_)}")
            elif hasattr(scaler_X, "n_features_in_"):
                print(f"  scaler_X.n_features_in_: {scaler_X.n_features_in_}")
            elif hasattr(scaler_X, "scale_") and scaler_X.scale_ is not None:
                print(f"  scaler_X.scale_ indicates {len(scaler_X.scale_)} features.")
            elif hasattr(scaler_X, "mean_") and scaler_X.mean_ is not None:
                print(f"  scaler_X.mean_ indicates {len(scaler_X.mean_)} features.")
            else:
                print("  Could not determine feature names or count from scaler_X attributes.")
        except Exception as e:
            print(f"  Error inspecting scaler_X: {e}")
    else:
        print("scaler_X is not loaded.")

    if scaler_y:
        print("Inspecting scaler_y:")
        try:
            if hasattr(scaler_y, "feature_names_in_"):
                print(f"  scaler_y.feature_names_in_ ({len(scaler_y.feature_names_in_)} features):")
                print(f"    {list(scaler_y.feature_names_in_)}")
            elif hasattr(scaler_y, "n_features_in_"):
                print(f"  scaler_y.n_features_in_: {scaler_y.n_features_in_}")
            elif hasattr(scaler_y, "scale_") and scaler_y.scale_ is not None:
                print(f"  scaler_y.scale_ indicates {len(scaler_y.scale_)} features.")
            elif hasattr(scaler_y, "mean_") and scaler_y.mean_ is not None:
                print(f"  scaler_y.mean_ indicates {len(scaler_y.mean_)} features.")
            else:
                print("  Could not determine feature names or count from scaler_y attributes.")
        except Exception as e:
            print(f"  Error inspecting scaler_y: {e}")
    else:
        print("scaler_y is not loaded.")
    print("--- End Inspecting Loaded Scalers ---\n")

    return scaler_X, scaler_y


def validate_scaler_coverage(
    scaler_X: Optional[object],
    scaler_y: Optional[object],
    trajectory_cols: Sequence[str],
    observation_cols: Sequence[str],
) -> None:
    """Emit coverage diagnostics indicating which columns can be inverse-scaled.

    - For scaler_X: verifies all scaler features are present in the available trajectory columns.
    - For scaler_y: verifies coverage and also compares feature sets vs. observation_cols.
    """
    # Validate scaler_X coverage
    if scaler_X and hasattr(scaler_X, "feature_names_in_"):
        scaler_X_feature_names = list(scaler_X.feature_names_in_)
        # Consider aliasing between 'time_delta' and 'next_visit_months'
        traj_set = set(trajectory_cols)
        if "time_delta" in traj_set:
            traj_set.add("next_visit_months")
        if "next_visit_months" in traj_set:
            traj_set.add("time_delta")
        missing_in_trajectory = [c for c in scaler_X_feature_names if c not in traj_set]
        if missing_in_trajectory:
            print("Warning: Some scaler_X features are not present in trajectory data and will not be unscaled:")
            print(f"  Missing: {missing_in_trajectory}")
        else:
            print(
                f"✓ scaler_X features validated: All {len(scaler_X_feature_names)} features expected by scaler_X are available in trajectory dataframes for unscaling."
            )

    # Validate scaler_y coverage vs model observation targets
    if scaler_y and hasattr(scaler_y, "feature_names_in_"):
        scaler_y_feature_names = list(scaler_y.feature_names_in_)
        traj_set = set(trajectory_cols)
        if "time_delta" in traj_set:
            traj_set.add("next_visit_months")
        if "next_visit_months" in traj_set:
            traj_set.add("time_delta")
        missing_in_trajectory = [c for c in scaler_y_feature_names if c not in traj_set]
        if not missing_in_trajectory:
            print(
                f"✓ scaler_y features validated: All {len(scaler_y_feature_names)} features expected by scaler_y are available in trajectory dataframes for unscaling."
            )
        else:
            print("Warning: Some scaler_y features are not present in trajectory data and will not be unscaled:")
            print(f"  Missing: {missing_in_trajectory}")

        if set(scaler_y_feature_names) != set(observation_cols):
            print("Warning: scaler_y features do not perfectly match observation_cols.")
            # Trimmed view for readability
            scaler_y_str = (
                f"List of {len(scaler_y_feature_names)} features, including '{scaler_y_feature_names[0]}', ..., '{scaler_y_feature_names[-1]}'"
                if scaler_y_feature_names
                else "[]"
            )
            observation_cols_str = (
                f"List of {len(observation_cols)} features, including '{observation_cols[0]}', ..., '{observation_cols[-1]}'"
                if observation_cols
                else "[]"
            )
            print(f"  scaler_y expects ({len(scaler_y_feature_names)}): {scaler_y_str}")
            print(f"  Model target features ({len(observation_cols)}): {observation_cols_str}")
            missing_in_scaler_y = sorted(list(set(observation_cols) - set(scaler_y_feature_names)))
            extra_in_scaler_y = sorted(list(set(scaler_y_feature_names) - set(observation_cols)))
            if missing_in_scaler_y:
                print(
                    f"    Features in model target but not in scaler_y (will not be unscaled by scaler_y): {missing_in_scaler_y}"
                )
            if extra_in_scaler_y:
                print(
                    f"    Features in scaler_y but not in model target (will be unscaled if present in data): {extra_in_scaler_y}"
                )
    elif scaler_y and not hasattr(scaler_y, "feature_names_in_"):
        print("Warning: scaler_y is loaded but does not have 'feature_names_in_'. Cannot validate its column structure.")


def inverse_scale_columns_inplace(
    df: pd.DataFrame,
    cols: List[str],
    scaler: Optional[object],
    scaler_name: str,
) -> None:
    """Inverse scale a subset of columns using per-feature StandardScaler stats.

    Columns not known to the scaler are ignored. The operation is in-place on df.
    """
    try:
        if scaler is None:
            print(f"{scaler_name} is not loaded; skipping inverse scaling for these columns: {cols}")
            return
        required_attrs = all(hasattr(scaler, attr) for attr in ["feature_names_in_", "scale_", "mean_"])
        if not required_attrs:
            print(f"Warning: {scaler_name} missing required attributes for column-wise inverse scaling. Skipping.")
            return
        scaler_feature_names = list(scaler.feature_names_in_)
        n_rows = len(df)
        n_features = len(scaler_feature_names)
        if n_rows == 0:
            return
        dense_matrix = np.zeros((n_rows, n_features), dtype=float)
        # Fill known columns from the dataframe; unknown remain zeros (mean in z-space)
        for feature_idx, feature_name in enumerate(scaler_feature_names):
            if feature_name in df.columns:
                dense_matrix[:, feature_idx] = df[feature_name].to_numpy(dtype=float)
            else:
                # Column not present; leave zeros which correspond to mean after inverse transform
                pass
        inverse_dense = (dense_matrix * scaler.scale_) + scaler.mean_
        # Write back only requested columns that are known to the scaler
        for feature_idx, feature_name in enumerate(scaler_feature_names):
            if feature_name in cols and feature_name in df.columns:
                df[feature_name] = inverse_dense[:, feature_idx]
    except Exception as e:
        print(f"Error during inverse scaling with {scaler_name}: {e}")


# --------------------------- Model / Indices ---------------------------

def build_category_indices(schema: Dict, observation_cols: Sequence[str]) -> Tuple[Dict[str, List[int]], List[int]]:
    """Return (group_indices, group_member_indices) for post-processing.

    - group_indices: mapping from group name to list of feature indices in observation_cols
    - group_member_indices: flattened, sorted list of all member indices across groups
    """
    y_categorical_groups = schema.get("y_categorical_groups", {}) or {}
    obs_index_map = {c: i for i, c in enumerate(observation_cols)}
    group_indices = {g: [obs_index_map[c] for c in cols if c in obs_index_map] for g, cols in y_categorical_groups.items()}
    # independent_bin_indices should be computed by the caller using bin_idx minus group members
    group_members = {i for idxs in group_indices.values() for i in idxs}
    return group_indices, sorted(list(group_members))


def postprocess_observation_vector(
    y_vec: torch.Tensor,
    observation_cols: Sequence[str],
    schema: Dict,
    ordinal_vars: Sequence[str],
    group_indices: Dict[str, List[int]],
    independent_bin_indices: List[int],
) -> torch.Tensor:
    """Apply rounding for ordinal vars, argmax within categorical groups, and threshold independent binaries.

    Returns a new tensor; does not modify the input tensor.
    """
    y_processed = y_vec.clone()
    # 1) Round ordinal continuous vars present in observation_cols
    y_continuous_feature_names = schema.get("y_cont_cols", [])
    ordinal_name_set = set(ordinal_vars)
    ordinal_columns = [c for c in y_continuous_feature_names if c in ordinal_name_set]
    obs_index_map = {c: i for i, c in enumerate(observation_cols)}
    ordinal_obs_indices = [obs_index_map[c] for c in ordinal_columns if c in obs_index_map]
    if ordinal_obs_indices:
        y_processed[ordinal_obs_indices] = torch.round(y_processed[ordinal_obs_indices])

    # 2) Enforce one-hot within categorical groups (argmax)
    for _, indices in group_indices.items():
        if not indices:
            continue
        group_values = y_processed[indices]
        winner = int(torch.argmax(group_values).item())
        y_processed[indices] = 0.0
        y_processed[indices[winner]] = 1.0

    # 3) Threshold independent binary columns at 0.5
    if independent_bin_indices:
        idx_tensor = torch.tensor(independent_bin_indices, dtype=torch.long, device=y_processed.device)
        y_processed[idx_tensor] = (y_processed[idx_tensor] >= 0.5).float()

    return y_processed


def autoregressive_rollout_for_subject(
    subject_true_df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    model_input_cols: Sequence[str],
    observation_cols: Sequence[str],
    state_cols: Sequence[str],
    action_cols: Sequence[str],
    time_col: str,
    time_delta_col: str,
    cont_idx: Sequence[int],
    bin_idx: Sequence[int],
    group_indices: Dict[str, List[int]],
    independent_bin_indices: List[int],
    ordinal_vars: Sequence[str],
    schema: Dict,
) -> List[Dict[str, float]]:
    """Run an autoregressive rollout for a single subject and return a list of state dicts."""
    predicted_state_dicts: List[Dict[str, float]] = []

    # Initial state
    current_state_values = subject_true_df.loc[0, state_cols].values
    initial_time_value = subject_true_df.loc[0, time_col]
    initial_state_map = dict(zip(state_cols, current_state_values))
    initial_state_map[time_col] = initial_time_value
    predicted_state_dicts.append(initial_state_map)

    input_sequence: List[List[float]] = []

    # Determine which time-gap feature name the model expects in its inputs.
    if time_delta_col in model_input_cols:
        gap_feature_name = time_delta_col
    elif "next_visit_months" in model_input_cols:
        gap_feature_name = "next_visit_months"
    elif "time_delta" in model_input_cols:
        gap_feature_name = "time_delta"
    else:
        gap_feature_name = None

    for step_idx in range(len(subject_true_df) - 1):
        time_delta_value = subject_true_df.loc[step_idx + 1, time_col] - subject_true_df.loc[step_idx, time_col]
        if time_delta_value <= 0:
            print(f"Warning: Non-positive time delta ({time_delta_value}) at step {step_idx}. Using 1.0.")
            time_delta_value = 1.0

        # Build input row in model_input_cols order
        current_state_map = dict(zip(state_cols, current_state_values))
        model_input_feature_map = current_state_map.copy()
        if gap_feature_name is not None:
            model_input_feature_map[gap_feature_name] = time_delta_value
        model_input_row = [model_input_feature_map[name] for name in model_input_cols]

        # Append the current state's input row BEFORE predicting, so last token matches step_idx
        input_sequence.append(model_input_row)

        x_seq_tensor = torch.tensor([input_sequence], dtype=torch.float32, device=device)
        seq_len = x_seq_tensor.size(1)
        key_padding_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

        with torch.no_grad():
            predicted_all_steps = model.infer_observations(
                x_seq_tensor,
                cont_idx=cont_idx,
                bin_idx=bin_idx,
                out_dim=len(observation_cols),
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                apply_sigmoid=True,
            )
            last_step_vector = predicted_all_steps[0, -1, :].detach()

        # Post-process predictions
        last_step_vector = postprocess_observation_vector(
            last_step_vector, observation_cols, schema, ordinal_vars, group_indices, independent_bin_indices
        )
        predicted_target_values = last_step_vector.cpu().numpy()

        # Build next state: predicted observations + true next actions
        next_state_map: Dict[str, float] = {}
        for i, feature_name in enumerate(observation_cols):
            next_state_map[feature_name] = float(predicted_target_values[i])
        true_next_action_series = subject_true_df.loc[step_idx + 1, action_cols]
        for feature_name in action_cols:
            next_state_map[feature_name] = float(true_next_action_series[feature_name])
        next_state_map[time_col] = float(subject_true_df.loc[step_idx + 1, time_col])

        # Save and advance
        predicted_state_dicts.append(next_state_map)
        current_state_values = np.array([next_state_map[name] for name in state_cols])

    return predicted_state_dicts
