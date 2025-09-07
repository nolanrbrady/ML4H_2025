"""
Expose preprocessing configuration at the package level.

This allows importing constants like:
    from autoreg.preprocessing import ORDINAL_VARS, CONTINUOUS_VARS
instead of reaching into the config module directly.
"""

from .config import (
    CONTINUOUS_VARS,
    ORDINAL_VARS,
    CATEGORICAL_VARS_FOR_IMPUTATION,
    DROP_VARS_FOR_IMPUTATION,
    ACTION_FEATURES,
    DRUG_CLASS_MAPPING,
    SCRIPT_DIR,
    ROOT_DIR,
    DATA_DIR,
    MODEL_TRAINING_DIR,
    CLINICIAN_POLICY_DIR,
    ALPACA_DIR,
)

__all__ = [
    "CONTINUOUS_VARS",
    "ORDINAL_VARS",
    "CATEGORICAL_VARS_FOR_IMPUTATION",
    "DROP_VARS_FOR_IMPUTATION",
    "ACTION_FEATURES",
    "DRUG_CLASS_MAPPING",
    "SCRIPT_DIR",
    "ROOT_DIR",
    "DATA_DIR",
    "MODEL_TRAINING_DIR",
    "CLINICIAN_POLICY_DIR",
    "ALPACA_DIR",
]
