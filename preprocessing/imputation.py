import joblib
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

# Fit on Train only

def _prepare_imputation_matrix(df, continuous_vars, ordinal_vars, categorical_vars_for_imputation_model, drop_vars):
    """Create the design matrix used by the imputer, with subject-wise means and OHE categoricals.
    Returns: (df_for_imputation, subject_mean_feature_names)
    """
    # Work on a copy to avoid mutating caller unexpectedly
    df = df.copy()

    # Drop variables NOT needed for imputation or as predictors (keep categoricals for OHE)
    vars_to_drop_immediately = [
        v for v in drop_vars if v not in categorical_vars_for_imputation_model
    ]
    df.drop(columns=vars_to_drop_immediately, inplace=True, errors='ignore')

    # Coerce numeric types for continuous & ordinal targets
    for col in continuous_vars + ordinal_vars:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Subject-specific mean features (help imputer respect within-subject structure) ---
    vars_for_subject_means = continuous_vars + ordinal_vars
    subject_mean_features_list = []
    if 'subject_id' in df.columns:
        for col in vars_for_subject_means:
            if col in df.columns:
                mean_col_name = f'{col}_subj_mean'
                df[mean_col_name] = df.groupby('subject_id')[col].transform('mean')
                subject_mean_features_list.append(mean_col_name)

    # Select columns for imputation matrix
    cols_to_select_for_imputation = (
        continuous_vars + ordinal_vars + subject_mean_features_list + categorical_vars_for_imputation_model
    )
    cols_to_select_for_imputation = [c for c in cols_to_select_for_imputation if c in df.columns]

    df_for_imputation = df[cols_to_select_for_imputation].copy()

    # One-hot encode categoricals used as predictors (do not create dummy_na to avoid leakage via NA flags)
    if categorical_vars_for_imputation_model:
        df_for_imputation = pd.get_dummies(
            df_for_imputation,
            columns=[c for c in categorical_vars_for_imputation_model if c in df_for_imputation.columns],
            prefix=[c for c in categorical_vars_for_imputation_model if c in df_for_imputation.columns],
            dummy_na=False,
        )

    return df_for_imputation, subject_mean_features_list


def train_mice_imputer(df_train,
                        continuous_vars,
                        ordinal_vars,
                        categorical_vars_for_imputation_model,
                        drop_vars):
    """Fit the IterativeImputer **only** on the training split and return the fitted imputer
    along with the exact training column schema used during fitting (for later alignment).
    """
    df_train_mat, _ = _prepare_imputation_matrix(
        df_train, continuous_vars, ordinal_vars, categorical_vars_for_imputation_model, drop_vars
    )

    mice_imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=30, random_state=0, n_jobs=-1),
        max_iter=50,
        random_state=0,
        sample_posterior=False,
        min_value=0,   # keep non-negative where appropriate
        verbose=2,
    )
    mice_imputer.fit(df_train_mat)

    # Persist the exact order/identity of training columns for downstream transform alignment
    training_schema = list(df_train_mat.columns)
    return mice_imputer, training_schema


def impute_with_trained_imputer(df,
                                imputer,
                                continuous_vars,
                                ordinal_vars,
                                categorical_vars_for_imputation_model,
                                drop_vars,
                                training_schema):
    """Apply a previously-fitted imputer to a (val/test) dataframe.
    Ensures categorical OHE alignment with the training schema and preserves ordinal rounding.
    Returns an imputed copy of `df`.
    """
    # Build matrix for this split
    df_mat, subject_mean_features_list = _prepare_imputation_matrix(
        df, continuous_vars, ordinal_vars, categorical_vars_for_imputation_model, drop_vars
    )

    # Align columns to the training schema: add missing as 0, drop extras, order exactly
    missing_cols = [c for c in training_schema if c not in df_mat.columns]
    for c in missing_cols:
        df_mat[c] = 0
    # Drop extras not seen during training
    extra_cols = [c for c in df_mat.columns if c not in training_schema]
    if extra_cols:
        df_mat.drop(columns=extra_cols, inplace=True)
    # Ensure exact column order
    df_mat = df_mat[training_schema]

    # Transform using the fitted imputer
    imputed_array = imputer.transform(df_mat)
    imputed_values_df = pd.DataFrame(imputed_array, columns=training_schema, index=df_mat.index)

    # Write imputed values back into a copy of the original df and keep OHE predictor columns
    out = df.copy()
    for col in imputed_values_df.columns:
        out[col] = imputed_values_df[col]

    # Round ordinals to preserve scale/type semantics
    for var in ordinal_vars:
        if var in out.columns:
            out[var] = out[var].round().astype(int)

    # Clean up temp subject-mean features (these are not intended as model inputs)
    out.drop(columns=[c for c in subject_mean_features_list if c in out.columns], inplace=True, errors='ignore')

    return out
