import pandas as pd
import numpy as np
from imputation import train_mice_imputer, impute_with_trained_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import joblib
import os
import json
from typing import List, Dict, Tuple, Set
from config import ALPACA_DIR, DRUG_CLASS_MAPPING, CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, DROP_VARS_FOR_IMPUTATION, MODEL_TRAINING_DIR

# =======================================================================================================================
# HELPER FUNCTIONS FOR PREPROCESSING
# =======================================================================================================================

def get_alpaca_observation_columns() -> List[str]:
    """
    Returns observation columns used for model (state) features.

    Behavior:
    - Binary demographics (OHE groups) are inferred from ALPACA `X_train.csv` if present,
      otherwise a stable fallback list is used to preserve prior behavior.
    - Continuous/ordinal observation columns are sourced dynamically from preprocessing
      configuration (CONTINUOUS_VARS + ORDINAL_VARS), so adding/removing variables in
      config propagates automatically without touching code. `months_since_bl` is kept
      separate but included in the returned list for downstream handling.
    """
    alpaca_training_path = os.path.join(ALPACA_DIR, 'X_train.csv')
    if os.path.exists(alpaca_training_path):
        alpaca_data = pd.read_csv(alpaca_training_path)
        binary_obs_cols = sorted([
            col for col in alpaca_data.columns
            if col.startswith(('PTGENDER_', 'PTRACCAT_', 'PTMARRY_'))
        ])
    else:
        binary_obs_cols = [
            'PTGENDER_Male', 'PTMARRY_Divorced', 'PTMARRY_Married', 'PTMARRY_Never married',
            'PTMARRY_Unknown', 'PTMARRY_Widowed', 'PTRACCAT_Asian', 'PTRACCAT_Black',
            'PTRACCAT_Hawaiian/Other PI', 'PTRACCAT_More than one', 'PTRACCAT_White'
        ]

    # Pull continuous + ordinal observation variables from config dynamically.
    # Exclude months_since_bl here; it is appended explicitly below and handled specially downstream.
    configured_numeric_obs = [
        v for v in (CONTINUOUS_VARS + ORDINAL_VARS)
        if v != 'months_since_bl'
    ]

    # Keep deterministic ordering: numeric obs follow config order; binary obs are sorted for stability.
    cont_ord_obs_cols = configured_numeric_obs

    # Final observation set includes months_since_bl as the final element for downstream processing.
    return cont_ord_obs_cols + binary_obs_cols + ['months_since_bl']

def map_drug_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Maps medication names to broader classes."""
    df['CMMED_clean'] = df['CMMED'].fillna('No Medication').astype(str).str.lower().str.strip()
    df['med_class'] = df['CMMED_clean'].map(DRUG_CLASS_MAPPING).fillna('Other')
    return df

def calculate_active_medications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates active medication flags for each subject-visit.

    This function determines which medications were active for a subject during any given visit.
    It handles the complexity of medication records not being tied to a specific visit and ensures
    that a visit is only flagged as 'No Medication' if no other medications were active.
    """
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'], errors='coerce')
    df['visit_year'] = df['EXAMDATE'].dt.year
    df['CMBGNYR_DRVD'] = pd.to_numeric(df['CMBGNYR_DRVD'], errors='coerce')
    # Fill end year with a distant future year for open-ended prescriptions
    df['CMENDYR_DRVD_filled'] = pd.to_numeric(df['CMENDYR_DRVD'], errors='coerce').fillna(9999)

    # Get unique visits to ensure each visit is processed once
    visit_info = df[['subject_id', 'visit', 'visit_year']].dropna().drop_duplicates()
    
    # Get unique medication records, excluding 'No Medication' which is handled separately
    medication_rows = df[
        (df['med_class'] != 'No Medication') & 
        (df['subject_id'].notna()) & 
        (df['CMBGNYR_DRVD'].notna()) & 
        (df['med_class'].notna())
    ][['subject_id', 'CMBGNYR_DRVD', 'CMENDYR_DRVD_filled', 'med_class']].drop_duplicates()

    # Cross-join every visit with every medication record for each subject.
    # This correctly applies a medication record to all of a subject's visits (past, present, future).
    expanded = pd.merge(visit_info, medication_rows, on='subject_id', how='left')
    
    # A medication is active if the visit year is within its start and end year
    active_mask = (expanded['visit_year'] >= expanded['CMBGNYR_DRVD']) & (expanded['visit_year'] <= expanded['CMENDYR_DRVD_filled'])
    expanded['is_active'] = active_mask.astype(int)

    # Pivot to create a column for each medication class with a `_active` flag
    active_meds_df = expanded.pivot_table(
        index=['subject_id', 'visit', 'visit_year'],
        columns='med_class',
        values='is_active',
        fill_value=0,
        aggfunc='max' # Use max to handle multiple entries of the same drug class for a visit
    ).reset_index()
    active_meds_df.columns.name = None # Clean up column index name

    # Ensure every visit is represented even if there were no med classes (empty pivot)
    # This guarantees we can set the 'No Medication_active' flag per-visit below.
    active_meds_df = visit_info.merge(active_meds_df, on=['subject_id', 'visit', 'visit_year'], how='left')

    # Rename columns to the '_active' format
    med_classes = sorted([col for col in active_meds_df.columns if col not in ['subject_id', 'visit', 'visit_year']])
    active_meds_df.rename(columns={mc: f'{mc}_active' for mc in med_classes}, inplace=True)

    # --- ROBUST 'No Medication' LOGIC ---
    # A visit is flagged 'No Medication' ONLY IF no other medications were active for that visit.
    # This prevents collisions and correctly implements the desired logic.
    all_med_cols = [f'{mc}_active' for mc in med_classes]
    if not all_med_cols: # Handle edge case where there are no medication columns at all
        active_meds_df['No Medication_active'] = 1
    else:
        active_meds_df['No Medication_active'] = (active_meds_df[all_med_cols].sum(axis=1) == 0).astype(int)
    
    # Merge the calculated active flags back into the original dataframe
    output_df = pd.merge(df, active_meds_df, on=['subject_id', 'visit', 'visit_year'], how='left')

    # Fill any NaNs that resulted from the merge with 0 (inactive) and ensure all columns exist
    final_active_cols = all_med_cols + ['No Medication_active']
    for col in final_active_cols:
        if col in output_df.columns:
            output_df[col] = output_df[col].fillna(0).astype(int)
        else:
            # If a med class had no active instances, the column might not exist. Add it as all zeros.
            output_df[col] = 0

    return output_df

def calculate_consistent_age(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamically computes a consistent age for each subject across all visits."""
    df['months_since_bl'] = pd.to_numeric(df['months_since_bl'], errors='coerce')
    df['subject_age'] = pd.to_numeric(df['subject_age'], errors='coerce')
    df.dropna(subset=['subject_id', 'months_since_bl', 'subject_age'], inplace=True)

    df_sorted = df.sort_values(['subject_id', 'months_since_bl'])
    earliest_visits = df_sorted.loc[df_sorted.groupby('subject_id')['months_since_bl'].idxmin()]
    
    earliest_visits['calculated_true_baseline_age'] = earliest_visits['subject_age'] - (earliest_visits['months_since_bl'] / 12.0)
    
    baseline_age_map = earliest_visits.set_index('subject_id')['calculated_true_baseline_age']
    df['calculated_true_baseline_age'] = df['subject_id'].map(baseline_age_map)
    
    df['subject_age'] = df['calculated_true_baseline_age'] + (df['months_since_bl'] / 12.0)
    df.drop(columns=['calculated_true_baseline_age'], inplace=True)
    
    return df

def filter_subjects_by_visit_count(df: pd.DataFrame, min_visits: int) -> pd.DataFrame:
    """Removes subjects with fewer than a specified number of visits."""
    visit_counts = df.groupby('subject_id').size()
    subjects_to_remove = visit_counts[visit_counts < min_visits].index
    return df[~df['subject_id'].isin(subjects_to_remove)]

def split_data_by_subject(df: pd.DataFrame, test_size: float, val_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Set, Set, Set]:
    """Splits data into train, validation, and test sets based on subject ID."""
    gss_main = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_main.split(df, groups=df['subject_id']))
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state + 1)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['subject_id']))
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    train_subjects = set(train_df['subject_id'].unique())
    val_subjects = set(val_df['subject_id'].unique())
    test_subjects = set(test_df['subject_id'].unique())

    return train_df, val_df, test_df, train_subjects, val_subjects, test_subjects

def impute_missing_values(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fits a MICE imputer on the training data and transforms all splits."""
    imputer, imputer_schema = train_mice_imputer(
        train_df.copy(), CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, DROP_VARS_FOR_IMPUTATION
    )
    
    imputed_train_df = impute_with_trained_imputer(
        train_df.copy(), imputer, CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, DROP_VARS_FOR_IMPUTATION, imputer_schema
    )
    imputed_val_df = impute_with_trained_imputer(
        val_df.copy(), imputer, CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, DROP_VARS_FOR_IMPUTATION, imputer_schema
    )
    imputed_test_df = impute_with_trained_imputer(
        test_df.copy(), imputer, CONTINUOUS_VARS, ORDINAL_VARS, CATEGORICAL_VARS_FOR_IMPUTATION, DROP_VARS_FOR_IMPUTATION, imputer_schema
    )
    
    joblib.dump({'schema_columns': imputer_schema}, os.path.join(MODEL_TRAINING_DIR, 'mice_imputer_schema.joblib'))
    
    return imputed_train_df, imputed_val_df, imputed_test_df

def align_columns(train_df, val_df, test_df):
    """Aligns columns of val and test dataframes to match the train dataframe."""
    train_cols = train_df.columns
    val_df = val_df.reindex(columns=train_cols, fill_value=0)
    test_df = test_df.reindex(columns=train_cols, fill_value=0)
    return train_df, val_df, test_df


def normalize_and_encode_sequence_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    categorical_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize numeric variables and one-hot encode categoricals for per-visit sequence datasets.

    - Preserves and returns DataFrames including `subject_id`.
    - Fits a StandardScaler on training numeric variables (CONTINUOUS_VARS + ORDINAL_VARS)
      present in the inputs and applies to val/test.
    - Applies one-hot encoding for `categorical_cols` present in each split, then aligns columns
      to the training split.
    - Saves the X scaler to MODEL_TRAINING_DIR/scaler_X.joblib and a Y scaler trained on
      observation numeric variables (CONTINUOUS_VARS + ORDINAL_VARS, if present) to
      MODEL_TRAINING_DIR/scaler_y.joblib for compatibility with downstream tooling.
    """
    # Preserve subject_id if present
    sid_train = train_df.get('subject_id', None)
    sid_val = val_df.get('subject_id', None)
    sid_test = test_df.get('subject_id', None)

    # Identify numeric variables present in the frames
    numeric_obs_vars = [v for v in (CONTINUOUS_VARS + ORDINAL_VARS) if v in train_df.columns]
    numeric_vars_all = list(numeric_obs_vars)
    # Include sequence-specific time gap feature if present (only for X scaling)
    for extra in ["next_visit_months", "time_since_prev", "time_delta"]:
        if extra in train_df.columns and extra not in numeric_vars_all:
            numeric_vars_all.append(extra)

    # Drop subject_id for transforms
    if 'subject_id' in train_df.columns:
        train_df = train_df.drop(columns=['subject_id'])
    if 'subject_id' in val_df.columns:
        val_df = val_df.drop(columns=['subject_id'])
    if 'subject_id' in test_df.columns:
        test_df = test_df.drop(columns=['subject_id'])

    # Fit scaler_y on ORIGINAL (unscaled) observation numeric variables BEFORE applying scaler_X
    # This preserves compatibility with downstream evaluation which expects scaler_y
    # to map model targets back to raw clinical units.
    scaler_y = StandardScaler()
    if numeric_obs_vars:
        scaler_y.fit(train_df[numeric_obs_vars])
        joblib.dump(scaler_y, os.path.join(MODEL_TRAINING_DIR, 'scaler_y.joblib'))

    # Fit/transform numeric columns for X (including time-gap if present)
    scaler_X = StandardScaler()
    if numeric_vars_all:
        train_df[numeric_vars_all] = scaler_X.fit_transform(train_df[numeric_vars_all])
        if val_df is not None and not val_df.empty:
            present = [c for c in numeric_vars_all if c in val_df.columns]
            if present:
                val_df[present] = scaler_X.transform(val_df[present])
        if test_df is not None and not test_df.empty:
            present = [c for c in numeric_vars_all if c in test_df.columns]
            if present:
                test_df[present] = scaler_X.transform(test_df[present])

    joblib.dump(scaler_X, os.path.join(MODEL_TRAINING_DIR, 'scaler_X.joblib'))

    # Determine categorical columns dynamically if not provided
    if categorical_cols is None:
        # Use config-driven list, but only include columns present in any split
        possible = list(CATEGORICAL_VARS_FOR_IMPUTATION)
        categorical_cols = [
            c for c in possible
            if (c in train_df.columns) or (c in val_df.columns) or (c in test_df.columns)
        ]

    # One-hot encode categoricals (only those present)
    def _encode(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in categorical_cols if c in df.columns]
        if not cols:
            return df
        return pd.get_dummies(df, columns=cols, prefix=cols, dummy_na=False, dtype=int)

    train_df = _encode(train_df)
    val_df = _encode(val_df)
    test_df = _encode(test_df)

    # Align val/test to train columns
    train_cols = list(train_df.columns)
    val_df = val_df.reindex(columns=train_cols, fill_value=0)
    test_df = test_df.reindex(columns=train_cols, fill_value=0)

    # Reattach subject_id where present
    if sid_train is not None:
        train_df.insert(0, 'subject_id', sid_train.values)
    if sid_val is not None:
        val_df.insert(0, 'subject_id', sid_val.values)
    if sid_test is not None:
        test_df.insert(0, 'subject_id', sid_test.values)

    # Note: scaler_y already fitted and saved above on ORIGINAL obs vars

    return train_df, val_df, test_df

def save_artifacts(data_dict: Dict[str, pd.DataFrame], schema: Dict, base_dir: str, schema_dirs: List[str]):
    """Saves dataframes and schema to specified locations."""
    os.makedirs(base_dir, exist_ok=True)
    for name, df in data_dict.items():
        df.to_csv(os.path.join(base_dir, f'{name}.csv'), index=False)
        print(f"Saved {name}.csv to {base_dir} with shape {df.shape}")

    for target_dir in schema_dirs:
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, 'columns_schema.json'), 'w') as f:
            json.dump(schema, f, indent=2)
    print(f"columns_schema.json written to: {', '.join(schema_dirs)}")

def normalize_clinician_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    save_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize clinician state features with a StandardScaler trained on train only.

    - Preserves and returns DataFrames including `subject_id` if present.
    - Fits a StandardScaler on all numeric feature columns present in the training split
      (excluding `subject_id`) and applies to val/test to avoid data leakage.
    - Saves the scaler to `{save_dir}/scaler_clinician_X.joblib`.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Preserve subject_id columns if present
    sid_train = X_train.get('subject_id', None)
    sid_val = X_val.get('subject_id', None)
    sid_test = X_test.get('subject_id', None)

    # Work on copies to avoid modifying inputs outside
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Drop subject_id for scaling work
    if 'subject_id' in X_train.columns:
        X_train.drop(columns=['subject_id'], inplace=True)
    if 'subject_id' in X_val.columns:
        X_val.drop(columns=['subject_id'], inplace=True)
    if 'subject_id' in X_test.columns:
        X_test.drop(columns=['subject_id'], inplace=True)

    # Determine numeric columns from training split
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        if not X_val.empty:
            present = [c for c in numeric_cols if c in X_val.columns]
            if present:
                X_val[present] = scaler.transform(X_val[present])
        if not X_test.empty:
            present = [c for c in numeric_cols if c in X_test.columns]
            if present:
                X_test[present] = scaler.transform(X_test[present])

    # Save scaler for downstream use
    joblib.dump(scaler, os.path.join(save_dir, 'scaler_clinician_X.joblib'))

    # Reattach subject_id if present
    if sid_train is not None:
        X_train.insert(0, 'subject_id', sid_train.values)
    if sid_val is not None:
        X_val.insert(0, 'subject_id', sid_val.values)
    if sid_test is not None:
        X_test.insert(0, 'subject_id', sid_test.values)

    return X_train, X_val, X_test


def transform_dataset(input_original_data: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function is tested and validated using a test datset in the data_exploration/reformat_data.py file.

    Transforms the original patient visit data into model_x (current state + time_delta)
    and model_y (next state) DataFrames.
    """
    model_x_list = []
    model_y_list = []

    # Calculate the total expected number of pairs
    total_expected_pairs = 0
    # We need to iterate over the groups to calculate this.
    # This will re-group if input_original_data is a DataFrame,
    # or iterate over an existing grouper if input_original_data was already grouped.
    # For robustness, let's assume input_original_data is the DataFrame.
    temp_grouped_for_check = input_original_data.groupby('subject_id')
    for _, subject_df_for_check in temp_grouped_for_check:
        num_subject_visits = len(subject_df_for_check)
        if num_subject_visits >= 2:
            total_expected_pairs += num_subject_visits * (num_subject_visits - 1) // 2

    # Group by subject for processing
    grouped_data = input_original_data.groupby('subject_id')

    for _, subject_visits_df in grouped_data:
        # Sort visits by time for each subject
        subject_visits_df = subject_visits_df.sort_values(by='months_since_bl').reset_index(drop=True)
        
        num_visits = len(subject_visits_df)
        if num_visits < 2:
            # Not enough visits to create a (current_state, next_state) pair
            continue

        # Iterate through visits to create (current_state, next_state) pairs
        # For each current visit, pair it with all subsequent visits
        for i in range(num_visits - 1):
            current_visit_series = subject_visits_df.iloc[i]
            for j in range(i + 1, num_visits):
                next_visit_series = subject_visits_df.iloc[j]

                time_delta = next_visit_series['months_since_bl'] - current_visit_series['months_since_bl']

                # Construct x_row data dictionary
                x_row_data = {'subject_id': current_visit_series['subject_id']}
                x_row_data.update(current_visit_series[feature_columns].to_dict())
                x_row_data['months_since_bl'] = current_visit_series['months_since_bl']
                x_row_data['time_delta'] = time_delta
                model_x_list.append(x_row_data)
                
                # Construct y_row data dictionary
                y_row_data = {'subject_id': next_visit_series['subject_id']}
                y_row_data.update(next_visit_series[feature_columns].to_dict())
                y_row_data['months_since_bl'] = next_visit_series['months_since_bl']
                model_y_list.append(y_row_data)

    # --- Integrated Check for Pair Counts ---
    actual_generated_pairs = len(model_x_list)
    if actual_generated_pairs != total_expected_pairs:
        # Detailed error message
        subject_counts_detail = input_original_data.groupby('subject_id').size()
        expected_pairs_detail = {}
        for sid, n_visits in subject_counts_detail.items():
            if n_visits >= 2:
                expected_pairs_detail[sid] = n_visits * (n_visits - 1) // 2
            else:
                expected_pairs_detail[sid] = 0
        
        # Check generated pairs per subject if possible (requires subject_id in model_x_list items)
        generated_pairs_per_subject_detail = {}
        if actual_generated_pairs > 0 and model_x_list and 'subject_id' in model_x_list[0]:
            temp_df_for_generated_counts = pd.DataFrame(model_x_list)
            generated_counts = temp_df_for_generated_counts['subject_id'].value_counts().to_dict()
            for sid in subject_counts_detail.keys(): # Iterate over all original subject_ids
                 generated_pairs_per_subject_detail[sid] = generated_counts.get(sid, 0)

        error_message = (
            f"Mismatch in generated pairs within transform_dataset. "
            f"Expected total pairs: {total_expected_pairs}, Got total pairs: {actual_generated_pairs}.\\n"
            f"Details per subject (Visits -> Expected Pairs | Generated Pairs):\\n"
        )
        for sid in subject_counts_detail.keys():
            expected = expected_pairs_detail.get(sid, "N/A")
            generated = generated_pairs_per_subject_detail.get(sid, "N/A")
            visits = subject_counts_detail.get(sid, "N/A")
            error_message += f"  Subject {sid}: ({visits} visits -> Expected: {expected} | Generated: {generated})\\n"

        raise ValueError(error_message)
    # --- End of Integrated Check ---

    # Define column order for model_x to match target_x.csv
    model_x_output_columns = ['subject_id'] + feature_columns + ['months_since_bl', 'time_delta']
    # Define column order for model_y to match target_y.csv
    model_y_output_columns = ['subject_id'] + feature_columns + ['months_since_bl']
    
    # Create DataFrames from the lists of dictionaries
    model_x = pd.DataFrame(model_x_list)
    model_y = pd.DataFrame(model_y_list)

    # Ensure correct column order and handle cases where no data pairs are generated
    if not model_x.empty:
        model_x = model_x[model_x_output_columns]
    else:
        # Create empty DataFrame with correct columns if no pairs were found
        model_x = pd.DataFrame(columns=model_x_output_columns)

    if not model_y.empty:
        model_y = model_y[model_y_output_columns] # Use new column order for model_y
    else:
        # Create empty DataFrame with correct columns if no pairs were found
        model_y = pd.DataFrame(columns=model_y_output_columns) # Use new column order for model_y
            
    return model_x, model_y


def transform_dataset_for_clinician(input_original_data: pd.DataFrame, feature_columns: list[str], action_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms the original patient visit data into clinician_x (current state)
    and clinician_y (current action) DataFrames for training a clinician policy model.
    
    This creates a direct mapping from patient state to clinical action: (state) -> (action)
    
    Args:
        input_original_data: DataFrame containing patient visit data
        feature_columns: List of feature column names (excluding actions)
        action_columns: List of action column names (e.g., drug "_active" columns)
    
    Returns:
        tuple: (clinician_x, clinician_y) DataFrames
    """
    clinician_x_list = []
    clinician_y_list = []

    # Calculate the total expected number of records (one per visit)
    total_expected_records = len(input_original_data)

    # Process each visit to create (state) -> (action) pairs
    for idx, visit_row in input_original_data.iterrows():
        # Construct clinician_x_row data dictionary (current state)
        x_row_data = {'subject_id': visit_row['subject_id']}
        # Add current state features
        x_row_data.update(visit_row[feature_columns].to_dict())
        # Do NOT add absolute time; clinician inputs exclude months_since_bl
        clinician_x_list.append(x_row_data)
        
        # Construct clinician_y_row data dictionary (current action)
        y_row_data = {'subject_id': visit_row['subject_id']}
        # Add current actions (what the clinician decided to prescribe at this visit)
        y_row_data.update(visit_row[action_columns].to_dict())
        clinician_y_list.append(y_row_data)

    # --- Validation Check ---
    actual_generated_records = len(clinician_x_list)
    if actual_generated_records != total_expected_records:
        error_message = (
            f"Mismatch in generated records within transform_dataset_for_clinician. "
            f"Expected total records: {total_expected_records}, Got total records: {actual_generated_records}."
        )
        raise ValueError(error_message)
    # --- End of Validation Check ---

    # Define column order for clinician_x (state features only; no absolute time)
    clinician_x_output_columns = ['subject_id'] + feature_columns
    # Define column order for clinician_y (actions only)
    clinician_y_output_columns = ['subject_id'] + action_columns
    
    # Create DataFrames from the lists of dictionaries
    clinician_x = pd.DataFrame(clinician_x_list)
    clinician_y = pd.DataFrame(clinician_y_list)

    # Ensure correct column order and handle cases where no data pairs are generated
    if not clinician_x.empty:
        clinician_x = clinician_x[clinician_x_output_columns]
    else:
        # Create empty DataFrame with correct columns if no pairs were found
        clinician_x = pd.DataFrame(columns=clinician_x_output_columns)

    if not clinician_y.empty:
        clinician_y = clinician_y[clinician_y_output_columns]
    else:
        # Create empty DataFrame with correct columns if no pairs were found
        clinician_y = pd.DataFrame(columns=clinician_y_output_columns)
            
    return clinician_x, clinician_y
