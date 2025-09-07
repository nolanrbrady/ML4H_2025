import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    map_drug_classes,
    calculate_active_medications,
    calculate_consistent_age,
    filter_subjects_by_visit_count,
    split_data_by_subject,
    align_columns,
    save_artifacts,
    transform_dataset,
    transform_dataset_for_clinician,
    get_alpaca_observation_columns,
    impute_missing_values
)
from imputation import train_mice_imputer, impute_with_trained_imputer

# Fixtures for sample data
@pytest.fixture
def sample_df():
    data = {
        'subject_id': [1, 1, 1, 2, 2, 3, 4, 4, 4, 4],
        'visit': [1, 2, 3, 1, 2, 1, 1, 2, 3, 4],
        'EXAMDATE': pd.to_datetime(['2020-01-01', '2020-06-01', '2021-01-01', '2020-02-01', '2021-02-01', '2020-03-01', '2020-04-01', '2020-10-01', '2021-04-01', '2021-10-01']),
        'CMMED': ['Aricept', 'Lipitor', 'Donepezil', 'Zocor', 'Aspirin', 'OtherMed', 'No Medication', 'Ibuprofen', 'Lexapro', 'Namenda'],
        'CMBGNYR_DRVD': [2020, 2020, 2021, 2020, 2020, 2020, np.nan, 2020, 2021, 2021],
        'CMENDYR_DRVD': [2020, 2020, 2021, 2020, 2020, 2022, np.nan, 2020, 2021, 2021],
        'months_since_bl': [0, 6, 12, 0, 12, 0, 0, 6, 12, 18],
        'subject_age': [65.0, 65.5, 66.0, 70.0, 71.0, 80.0, 75.0, 75.5, 76.0, 76.5],
        'feature1': [10, 12, 11, 20, 22, 30, 40, 42, 41, 43],
        'PTGENDER': ['Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Female'],
        'PTRACCAT': ['White', 'White', 'White', 'Black', 'Black', 'Asian', 'White', 'White', 'White', 'White'],
        'research_group': ['CN', 'CN', 'CN', 'AD', 'AD', 'MCI', 'CN', 'CN', 'CN', 'CN']
    }
    return pd.DataFrame(data)

@pytest.fixture
def imputation_test_data():
    """Provides sample data for testing the imputation functions."""
    train_df = pd.DataFrame({
        'subject_id': [1, 1, 2, 2],
        'continuous_var': [10.0, 20.0, np.nan, 40.0],
        'ordinal_var': [1, np.nan, 3, 4],
        'PTGENDER': ['Male', 'Female', 'Male', 'Female'],
        'research_group': ['CN', 'AD', 'CN', 'AD']
    })
    val_df = pd.DataFrame({
        'subject_id': [3, 3],
        'continuous_var': [50.0, np.nan],
        'ordinal_var': [5, 6],
        'PTGENDER': ['Male', 'Female'],
        'research_group': ['MCI', 'MCI']
    })
    return train_df, val_df

def test_imputation_functions(imputation_test_data):
    """
    Tests the train_mice_imputer and impute_with_trained_imputer functions.
    This is an integration test that checks if the imputation process works end-to-end.
    """
    train_df, val_df = imputation_test_data
    
    # Define column types for the imputation
    continuous_vars = ['continuous_var']
    ordinal_vars = ['ordinal_var']
    categorical_vars = ['PTGENDER', 'research_group']
    drop_vars = []

    # 1. Test training the imputer
    imputer, schema = train_mice_imputer(
        train_df, continuous_vars, ordinal_vars, categorical_vars, drop_vars
    )
    assert imputer is not None
    assert isinstance(schema, list)

    # 2. Test imputing with the trained imputer
    imputed_val_df = impute_with_trained_imputer(
        val_df, imputer, continuous_vars, ordinal_vars, categorical_vars, drop_vars, schema
    )

    # Assertions
    assert not imputed_val_df['continuous_var'].isnull().any(), "Continuous column should be imputed."
    assert pd.api.types.is_integer_dtype(imputed_val_df['ordinal_var']), "Ordinal column should be rounded to integer."
    assert imputed_val_df.shape[0] == val_df.shape[0], "Number of rows should be preserved."

def test_map_drug_classes(sample_df):
    processed_df = map_drug_classes(sample_df)
    assert 'med_class' in processed_df.columns
    assert processed_df.loc[0, 'med_class'] == 'AD Treatment'
    assert processed_df.loc[1, 'med_class'] == 'Statin'
    assert processed_df.loc[5, 'med_class'] == 'Other'
    assert processed_df.loc[6, 'med_class'] == 'No Medication'

def test_calculate_active_medications(sample_df):
    df = map_drug_classes(sample_df)
    processed_df = calculate_active_medications(df)
    active_cols = [col for col in processed_df.columns if '_active' in col]
    assert len(active_cols) > 0
    assert processed_df.loc[0, 'AD Treatment_active'] == 1
    assert processed_df.loc[2, 'AD Treatment_active'] == 1
    assert processed_df.loc[3, 'Statin_active'] == 1
    # For the visit at index 6, 'No Medication_active' should be 0 because another
    # medication ('Ibuprofen') was active for the same subject in the same year.
    assert processed_df.loc[6, 'No Medication_active'] == 0

def test_calculate_consistent_age(sample_df):
    df = sample_df.copy()
    df.loc[1, 'subject_age'] = 68.0 # Introduce inconsistency
    processed_df = calculate_consistent_age(df)
    ages_subject1 = processed_df[processed_df['subject_id'] == 1]['subject_age']
    baseline_age = ages_subject1.iloc[0]
    assert np.isclose(ages_subject1.iloc[1], baseline_age + 0.5)
    assert np.isclose(ages_subject1.iloc[2], baseline_age + 1.0)

def test_filter_subjects_by_visit_count(sample_df):
    processed_df = filter_subjects_by_visit_count(sample_df, min_visits=3)
    assert 1 in processed_df['subject_id'].unique()
    assert 4 in processed_df['subject_id'].unique()
    assert 2 not in processed_df['subject_id'].unique()
    assert 3 not in processed_df['subject_id'].unique()

def test_split_data_by_subject(sample_df):
    train_df, val_df, test_df, train_subjects, val_subjects, test_subjects = split_data_by_subject(
        sample_df, test_size=0.4, val_size=0.2, random_state=42
    )
    assert len(train_subjects.intersection(val_subjects)) == 0
    assert len(train_subjects.intersection(test_subjects)) == 0
    assert len(val_subjects.intersection(test_subjects)) == 0

def test_align_columns():
    train_df = pd.DataFrame({'a': [1], 'b': [2]})
    val_df = pd.DataFrame({'a': [3], 'c': [4]})
    test_df = pd.DataFrame({'b': [5]})
    _, aligned_val, aligned_test = align_columns(train_df, val_df, test_df)
    assert list(aligned_val.columns) == ['a', 'b']
    assert aligned_val.loc[0, 'a'] == 3
    assert aligned_val.loc[0, 'b'] == 0
    assert list(aligned_test.columns) == ['a', 'b']
    assert aligned_test.loc[0, 'a'] == 0
    assert aligned_test.loc[0, 'b'] == 5

def test_transform_dataset_subject_pairing_count():
    """Tests if transform_dataset generates the correct number of pairs."""
    data = {
        'subject_id': [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        'months_since_bl': [0, 6, 12, 18, 24, 0, 6, 12, 0, 6, 0],
        'feature1': [10, 12, 11, 13, 14, 20, 22, 21, 30, 31, 40],
        'feature2': [1, 2, 1, 3, 2, 5, 6, 5, 8, 7, 9]
    }
    input_df = pd.DataFrame(data)
    feature_columns = ['feature1', 'feature2']
    expected_pairs = (5 * 4 // 2) + (3 * 2 // 2) + (2 * 1 // 2) + 0
    model_x, model_y = transform_dataset(input_df, feature_columns)
    assert len(model_x) == expected_pairs
    assert len(model_y) == expected_pairs
    assert 'time_delta' in model_x.columns

def test_transform_dataset_for_clinician():
    """Tests the clinician data transformation."""
    data = {
        'subject_id': [1, 1, 1, 2, 2],
        'months_since_bl': [0, 6, 12, 0, 6],
        'state1': [10, 11, 12, 20, 21],
        'action1_active': [0, 1, 0, 1, 0]
    }
    input_df = pd.DataFrame(data)
    state_cols = ['state1']
    action_cols = ['action1_active']
    x_df, y_df = transform_dataset_for_clinician(input_df, state_cols, action_cols)
    assert len(x_df) == len(input_df)
    assert len(y_df) == len(input_df)
    assert 'action1_active' in y_df.columns
    assert 'state1' in x_df.columns
    assert 'months_since_bl' in x_df.columns

def test_save_artifacts(tmp_path):
    """Tests saving of dataframes and schema."""
    data_dict = {'test_df': pd.DataFrame({'a': [1]})}
    schema = {'key': 'value'}
    base_dir = tmp_path / 'output'
    schema_dir = tmp_path / 'schema'
    save_artifacts(data_dict, schema, str(base_dir), [str(schema_dir)])
    assert os.path.exists(base_dir / 'test_df.csv')
    assert os.path.exists(schema_dir / 'columns_schema.json')


def test_preprocess_main_smoke_small_csv(tmp_path, monkeypatch):
    """
    Smoke test: run preprocess.main on a tiny synthetic ADNI_merged.csv with
    monkeypatched directories, mappings, and stubs for splitting and imputation/encoding.
    Verifies that expected artifacts and schema are written and that key columns exist.
    """
    import preprocess as preprocess_mod
    import utils as utils_mod

    # 1) Prepare temp directories
    data_dir = tmp_path / "data"
    model_training_dir = tmp_path / "model_training"
    clinician_dir = tmp_path / "clinician"
    alpaca_dir = tmp_path / "alpaca"
    for d in [data_dir, model_training_dir, clinician_dir, alpaca_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2) Create a tiny synthetic ADNI_merged.csv (3 subjects, 3 visits each)
    #    Include minimal columns used by the pipeline
    df = pd.DataFrame({
        'subject_id': [101, 101, 101, 202, 202, 202, 303, 303, 303],
        'visit':      [1,   2,   3,   1,   2,   3,   1,   2,   3],
        'EXAMDATE':   pd.to_datetime([
            '2020-01-01', '2020-07-01', '2021-01-01',
            '2020-02-01', '2020-08-01', '2021-02-01',
            '2020-03-01', '2020-09-01', '2021-03-01',
        ]),
        'CMMED': [
            'Aricept', None, 'Aricept',
            'Lipitor', None, 'Lipitor',
            None, None, None,
        ],
        # Start/End years only for actual meds
        'CMBGNYR_DRVD': [2020, np.nan, 2020, 2020, np.nan, 2020, np.nan, np.nan, np.nan],
        'CMENDYR_DRVD': [2021, np.nan, 2021, 2020, np.nan, 2021, np.nan, np.nan, np.nan],
        'months_since_bl': [0, 6, 12, 0, 6, 12, 0, 6, 12],
        # Ages will be recomputed by calculate_consistent_age; seed them close to consistent
        'subject_age': [70.0, 70.5, 71.0, 75.0, 75.5, 76.0, 65.0, 65.5, 66.0],
        'PTGENDER': ['Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male'],
        'PTRACCAT': ['White', 'White', 'White', 'Black', 'Black', 'Black', 'White', 'White', 'White'],
        'research_group': ['CN', 'CN', 'CN', 'AD', 'AD', 'AD', 'MCI', 'MCI', 'MCI'],
    })
    csv_path = data_dir / 'ADNI_merged.csv'
    df.to_csv(csv_path, index=False)

    # 3) Monkeypatch constants to local temp dirs and minimal configs
    monkeypatch.setattr(preprocess_mod, 'DATA_DIR', str(data_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    # Also update utils' MODEL_TRAINING_DIR so scalers save to the same place
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'CLINICIAN_POLICY_DIR', str(clinician_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'ALPACA_DIR', str(alpaca_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'COLUMNS_TO_DROP', [], raising=False)

    # Minimal variables for downstream logic
    monkeypatch.setattr(preprocess_mod, 'CONTINUOUS_VARS', ['subject_age'], raising=False)
    monkeypatch.setattr(preprocess_mod, 'ORDINAL_VARS', [], raising=False)
    # Keep utils in sync for scaling selections
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['subject_age', 'months_since_bl'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', [], raising=False)

    # Action features: ensure No Medication and one AD treatment are present
    monkeypatch.setattr(preprocess_mod, 'ACTION_FEATURES', ['AD Treatment_active', 'No Medication_active'], raising=False)

    # Map a few meds and make sure 'no medication' is explicitly mapped
    monkeypatch.setattr(utils_mod, 'DRUG_CLASS_MAPPING', {
        'aricept': 'AD Treatment',
        'lipitor': 'Statin',
        'no medication': 'No Medication',
    }, raising=False)

    # 4) Stub functions inside preprocess to reduce complexity and make the test deterministic
    def fake_split_data_by_subject(df_in, test_size, val_size, random_state):
        subs = sorted(df_in['subject_id'].unique())
        assert len(subs) >= 3, 'Smoke test expects at least 3 unique subjects'
        train_subjects, val_subjects, test_subjects = {subs[0]}, {subs[1]}, {subs[2]}
        train_df = df_in[df_in['subject_id'].isin(train_subjects)].copy()
        val_df = df_in[df_in['subject_id'].isin(val_subjects)].copy()
        test_df = df_in[df_in['subject_id'].isin(test_subjects)].copy()
        return train_df, val_df, test_df, train_subjects, val_subjects, test_subjects

    def fake_get_alpaca_observation_columns():
        # Keep tiny and predictable; missing columns will be injected as zeros where needed
        return ['subject_age', 'PTGENDER_Male', 'PTRACCAT_White', 'months_since_bl']

    def fake_impute_missing_values(train_df, val_df, test_df):
        # Identity imputation to keep things simple
        return train_df.copy(), val_df.copy(), test_df.copy()

    def fake_normalize_and_encode(X_train, X_val, X_test, y_train, y_val, y_test):
        # Pass-through; we only care about pipeline orchestration and outputs existing
        return X_train, X_val, X_test, y_train, y_val, y_test

    monkeypatch.setattr(preprocess_mod, 'split_data_by_subject', fake_split_data_by_subject, raising=False)
    monkeypatch.setattr(preprocess_mod, 'get_alpaca_observation_columns', fake_get_alpaca_observation_columns, raising=False)
    monkeypatch.setattr(preprocess_mod, 'impute_missing_values', fake_impute_missing_values, raising=False)
    monkeypatch.setattr(preprocess_mod, 'normalize_and_encode', fake_normalize_and_encode, raising=False)

    # 5) Run the pipeline
    preprocess_mod.main()

    # 6) Assert artifacts exist
    # Model artifacts
    for name in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        assert os.path.exists(model_training_dir / f'{name}.csv'), f"Missing {name}.csv"

    # Clinician policy artifacts
    for name in [
        'clinician_X_train', 'clinician_X_val', 'clinician_X_test',
        'clinician_y_train', 'clinician_y_val', 'clinician_y_test',
    ]:
        assert os.path.exists(clinician_dir / f'{name}.csv'), f"Missing {name}.csv in clinician dir"

    # Schema saved to all target dirs
    for d in [model_training_dir, alpaca_dir, clinician_dir]:
        assert os.path.exists(d / 'columns_schema.json'), f"Missing schema in {d}"

    # 7) Spot-check contents
    X_train_df = pd.read_csv(model_training_dir / 'X_train.csv')
    y_train_df = pd.read_csv(model_training_dir / 'y_train.csv')
    X_val_df = pd.read_csv(model_training_dir / 'X_val.csv')
    y_val_df = pd.read_csv(model_training_dir / 'y_val.csv')

    # X should include time info
    assert 'months_since_bl' in X_train_df.columns
    assert 'time_delta' in X_train_df.columns

    # y should not include action "_active" columns after cleanup
    assert not any(c.endswith('_active') for c in y_train_df.columns)

    # subject_id handling: present only in validation splits
    assert 'subject_id' not in X_train_df.columns
    assert 'subject_id' not in y_train_df.columns
    assert 'subject_id' in X_val_df.columns
    assert 'subject_id' in y_val_df.columns

    # Shapes should be non-empty given our deterministic split
    assert len(X_train_df) > 0
    assert len(y_train_df) > 0

    # 8) Validate schema contents
    import json
    with open(model_training_dir / 'columns_schema.json', 'r') as f:
        schema = json.load(f)

    # Action columns should include our stubbed actions
    assert 'action_cols' in schema and isinstance(schema['action_cols'], list)
    assert 'AD Treatment_active' in schema['action_cols']
    assert 'No Medication_active' in schema['action_cols']

    # Continuous observation columns should include the stubbed continuous var
    assert 'cont_obs_cols' in schema and 'subject_age' in schema['cont_obs_cols']

    # Model input columns include temporal features
    assert 'model_input_cols' in schema
    assert 'months_since_bl' in schema['model_input_cols']
    assert 'time_delta' in schema['model_input_cols']

def test_calculate_active_medications_no_meds_all_visits():

    df = pd.DataFrame({
        "subject_id": [10, 10],
        "visit": [1, 2],
        "EXAMDATE": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "CMMED": [None, None],
        "CMBGNYR_DRVD": [None, None],
        "CMENDYR_DRVD": [None, None]
    })
    df = map_drug_classes(df)
    out = calculate_active_medications(df)
    assert "No Medication_active" in out.columns
    assert out["No Medication_active"].tolist() == [1, 1]

def test_calculate_consistent_age_drops_invalid_and_aligns():

    df = pd.DataFrame({
        "subject_id": [1, 1, 1, 2],
        "months_since_bl": [0, 6, np.nan, 0],  # one invalid row
        "subject_age": [70.0, 70.5, 71.0, np.nan],  # one invalid row
    })
    out = calculate_consistent_age(df)
    # Invalid rows dropped => keep (1,0), (1,6)
    assert set(map(tuple, out[["subject_id","months_since_bl"]].values)) == {(1,0),(1,6)}
    # Consistency: 6 months later = +0.5 years
    s1 = out[out.subject_id == 1].sort_values("months_since_bl")["subject_age"].tolist()
    assert len(s1) == 2 and abs((s1[1] - s1[0]) - 0.5) < 1e-6


def test_normalize_and_encode_subject_id_and_one_hot(tmp_path, monkeypatch):
    import pandas as pd
    import numpy as np
    import utils as utils_mod

    # Force which columns get scaled
    monkeypatch.setattr(utils_mod, "CONTINUOUS_VARS", ["z1"])
    monkeypatch.setattr(utils_mod, "ORDINAL_VARS", ["z2"])

    # Build splits with subject_id in val only, per contract in preprocess.main
    X_train = pd.DataFrame({"z1":[0.0, 2.0], "z2":[1.0, 3.0], "time_delta":[1.0, 2.0], "PTGENDER":["Male","Female"], "PTRACCAT":["White","Black"]})
    X_val   = pd.DataFrame({"subject_id":[10,11], "z1":[1.0, 1.0], "z2":[2.0, 2.0], "time_delta":[1.5, 1.5], "PTGENDER":["Male","Male"], "PTRACCAT":["White","White"]})
    X_test  = pd.DataFrame({"z1":[4.0], "z2":[5.0], "time_delta":[3.0], "PTGENDER":["Female"], "PTRACCAT":["Black"]})

    y_train = pd.DataFrame({"z1":[10.0, 20.0], "z2":[1.0, 2.0]})
    y_val   = pd.DataFrame({"subject_id":[10,11], "z1":[15.0, 25.0], "z2":[1.5, 2.5]})
    y_test  = pd.DataFrame({"z1":[30.0], "z2":[3.0]})

    # Prevent scalers from being written to the real project directory
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(tmp_path), raising=False)

    X_tr, X_va, X_te, y_tr, y_va, y_te = utils_mod.normalize_and_encode(X_train, X_val, X_test, y_train, y_val, y_test)

    # subject_id only preserved in val splits
    assert "subject_id" not in X_tr.columns and "subject_id" in X_va.columns
    assert "subject_id" not in y_tr.columns and "subject_id" in y_va.columns

    # One-hot encoding applied (original categorical cols removed)
    assert "PTGENDER" not in X_tr.columns and any(c.startswith("PTGENDER_") for c in X_tr.columns)
    assert "PTRACCAT" not in X_tr.columns and any(c.startswith("PTRACCAT_") for c in X_tr.columns)

    # Scaling happened for X_train: z1, z2, time_delta centered near 0
    assert abs(X_tr["z1"].mean()) < 1e-6
    assert abs(X_tr["z2"].mean()) < 1e-6
    assert abs(X_tr["time_delta"].mean()) < 1e-6

def test_transform_dataset_time_delta_and_alignment():
    import pandas as pd
    from utils import transform_dataset
    df = pd.DataFrame({
        "subject_id":[1,1,1],
        "months_since_bl":[0,6,12],
        "f":[10,11,12]
    })
    X, Y = transform_dataset(df, ["f"])
    # pairs: (0->6),(0->12),(6->12)
    assert X.shape[0] == 3 and Y.shape[0] == 3
    # Check time_deltas exactly
    assert X["time_delta"].tolist() == [6, 12, 6]
    # Check alignment of months_since_bl
    assert X["months_since_bl"].tolist() == [0, 0, 6]
    assert Y["months_since_bl"].tolist() == [6, 12, 12]

def test_transform_dataset_for_clinician_maps_actions():
    import pandas as pd
    from utils import transform_dataset_for_clinician
    df = pd.DataFrame({
        "subject_id":[1,1],
        "months_since_bl":[0,6],
        "s":[5,6],
        "A_active":[0,1],
    })
    X, Y = transform_dataset_for_clinician(df, ["s"], ["A_active"])
    assert X["s"].tolist() == [5,6]
    assert Y["A_active"].tolist() == [0,1]


def test_impute_missing_values_wrapper_saves_schema(tmp_path, monkeypatch):
    import utils as utils_mod
    import numpy as np
    import pandas as pd

    # Narrow the variables to those present in our toy data
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['cont'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', ['ord'], raising=False)
    monkeypatch.setattr(utils_mod, 'CATEGORICAL_VARS_FOR_IMPUTATION', ['PTGENDER', 'research_group'], raising=False)
    monkeypatch.setattr(utils_mod, 'DROP_VARS_FOR_IMPUTATION', [], raising=False)
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(tmp_path), raising=False)

    # Build simple splits with NaNs
    train_df = pd.DataFrame({
        'subject_id': [1,1,2,2],
        'cont': [1.0, np.nan, 3.0, 4.0],
        'ord': [1, 2, np.nan, 4],
        'PTGENDER': ['Male','Female','Male','Female'],
        'research_group': ['CN','AD','CN','AD']
    })
    val_df = pd.DataFrame({
        'subject_id': [3,3],
        'cont': [np.nan, 6.0],
        'ord': [2, 3],
        'PTGENDER': ['Male','Male'],
        'research_group': ['MCI','MCI']
    })
    test_df = pd.DataFrame({
        'subject_id': [4],
        'cont': [np.nan],
        'ord': [np.nan],
        'PTGENDER': ['Female'],
        'research_group': ['CN']
    })

    imputed_train, imputed_val, imputed_test = utils_mod.impute_missing_values(train_df, val_df, test_df)

    # No NaNs should remain in imputed targets
    for df_out in [imputed_train, imputed_val, imputed_test]:
        assert not df_out['cont'].isna().any()
        assert not df_out['ord'].isna().any()
        # Ordinal should be ints after rounding
        assert pd.api.types.is_integer_dtype(df_out['ord'])

    # Schema file saved
    assert (tmp_path / 'mice_imputer_schema.joblib').exists()


def test_get_alpaca_observation_columns_default(tmp_path, monkeypatch):
    import utils as utils_mod

    # Point ALPACA_DIR to an empty temp dir so fallback list is used
    alp_dir = tmp_path / 'alp'
    alp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(utils_mod, 'ALPACA_DIR', str(alp_dir), raising=False)

    cols = utils_mod.get_alpaca_observation_columns()
    # Ensure key expected defaults are present
    assert 'months_since_bl' in cols
    assert 'PTGENDER_Male' in cols

def test_get_alpaca_observation_columns_from_file(tmp_path, monkeypatch):
    import utils as utils_mod
    # Create an ALPACA_DIR with a minimal X_train.csv including only a subset of OHE binaries
    alp_dir = tmp_path / 'alp_from_file'
    alp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(utils_mod, 'ALPACA_DIR', str(alp_dir), raising=False)

    # Minimal X_train with the needed columns present
    pd.DataFrame({
        'TAU_data': [0.0],
        'subject_age': [70.0],
        'months_since_bl': [0],
        'PTGENDER_Male': [1],
        'PTRACCAT_White': [1],
    }).to_csv(alp_dir / 'X_train.csv', index=False)

    cols = utils_mod.get_alpaca_observation_columns()
    # Should include continuous defaults and only the present OHE binaries from the file
    assert 'PTGENDER_Male' in cols and 'PTRACCAT_White' in cols
    assert 'PTRACCAT_Black' not in cols  # Not present in file
    assert 'months_since_bl' in cols


def test_normalize_and_encode_unseen_category_alignment(tmp_path, monkeypatch):
    import utils as utils_mod
    # Restrict which variables get scaled to make this predictable
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['numeric_feature'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', [], raising=False)
    # Prevent scalers from being written to the real project directory
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(tmp_path), raising=False)

    train_features = pd.DataFrame({
        'numeric_feature': [0.0, 1.0],
        'time_delta': [1.0, 2.0],
        'PTGENDER': ['Male', 'Male'],  # Train sees only 'Male'
        'PTRACCAT': ['White', 'White'],
    })
    val_features = pd.DataFrame({
        'subject_id': [1, 2],
        'numeric_feature': [2.0, 3.0],
        'time_delta': [1.5, 1.5],
        'PTGENDER': ['Female', 'Female'],  # Unseen category
        'PTRACCAT': ['Black', 'Black'],
    })
    test_features = pd.DataFrame({
        'numeric_feature': [4.0],
        'time_delta': [2.5],
        'PTGENDER': ['Female'],
        'PTRACCAT': ['Black'],
    })
    train_targets = pd.DataFrame({'numeric_feature': [10.0, 20.0]})
    val_targets = pd.DataFrame({'subject_id': [1, 2], 'numeric_feature': [15.0, 25.0]})
    test_targets = pd.DataFrame({'numeric_feature': [30.0]})

    processed_X_train, processed_X_val, processed_X_test, processed_y_train, processed_y_val, processed_y_test = utils_mod.normalize_and_encode(
        train_features.copy(), val_features.copy(), test_features.copy(),
        train_targets.copy(), val_targets.copy(), test_targets.copy()
    )

    # After OHE and align to train, val/test should not contain unseen category columns
    assert not any(c.startswith('PTGENDER_') and 'Female' in c for c in processed_X_val.columns)
    assert any(c == 'PTGENDER_Male' for c in processed_X_val.columns)
    # Rows that were Female should have PTGENDER_Male == 0
    if 'PTGENDER_Male' in processed_X_val.columns:
        assert (processed_X_val['PTGENDER_Male'] == 0).all()
    # subject_id preserved in val
    assert 'subject_id' in processed_X_val.columns and 'subject_id' in processed_y_val.columns


def test_preprocess_main_schema_consistency(tmp_path, monkeypatch):
    import preprocess as preprocess_mod
    import utils as utils_mod
    import json
    import joblib

    # Setup directories
    data_dir = tmp_path / 'data2'
    model_training_dir = tmp_path / 'mt2'
    clinician_dir = tmp_path / 'clin2'
    alpaca_dir = tmp_path / 'alp2'
    for d in [data_dir, model_training_dir, clinician_dir, alpaca_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Minimal ADNI_merged.csv
    df = pd.DataFrame({
        'subject_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'visit': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'EXAMDATE': pd.to_datetime([
            '2020-01-01','2020-07-01','2021-01-01',
            '2020-02-01','2020-08-01','2021-02-01',
            '2020-03-01','2020-09-01','2021-03-01',
        ]),
        'CMMED': ['Aricept', None, None, 'No Medication', None, None, 'Lipitor', None, None],
        'CMBGNYR_DRVD': [2020, np.nan, np.nan, np.nan, np.nan, np.nan, 2020, np.nan, np.nan],
        'CMENDYR_DRVD': [2021, np.nan, np.nan, np.nan, np.nan, np.nan, 2021, np.nan, np.nan],
        'months_since_bl': [0, 6, 12, 0, 6, 12, 0, 6, 12],
        'subject_age': [70.0, 70.5, 71.0, 75.0, 75.5, 76.0, 65.0, 65.5, 66.0],
        'PTGENDER': ['Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male'],
        'PTRACCAT': ['White', 'White', 'White', 'Black', 'Black', 'Black', 'White', 'White', 'White'],
        'research_group': ['CN', 'CN', 'CN', 'AD', 'AD', 'AD', 'MCI', 'MCI', 'MCI'],
    })
    df.to_csv(data_dir / 'ADNI_merged.csv', index=False)

    # Configure preprocess to use our dirs and minimal settings
    monkeypatch.setattr(preprocess_mod, 'DATA_DIR', str(data_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    # Ensure utils writes scalers to the temp model_training_dir as well
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(model_training_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'CLINICIAN_POLICY_DIR', str(clinician_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'ALPACA_DIR', str(alpaca_dir), raising=False)
    monkeypatch.setattr(preprocess_mod, 'COLUMNS_TO_DROP', [], raising=False)
    monkeypatch.setattr(preprocess_mod, 'CONTINUOUS_VARS', ['subject_age'], raising=False)
    monkeypatch.setattr(preprocess_mod, 'ORDINAL_VARS', [], raising=False)
    # Ensure utils scales expected targets as well
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['subject_age', 'months_since_bl'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', [], raising=False)
    monkeypatch.setattr(preprocess_mod, 'ACTION_FEATURES', ['AD Treatment_active', 'No Medication_active'], raising=False)

    # Keep observation columns minimal to ensure continuity through the pipeline
    monkeypatch.setattr(preprocess_mod, 'get_alpaca_observation_columns', lambda: ['subject_age', 'months_since_bl'], raising=False)

    # Run full pipeline
    preprocess_mod.main()

    # Load artifacts and schema
    X_train = pd.read_csv(model_training_dir / 'X_train.csv')
    X_val = pd.read_csv(model_training_dir / 'X_val.csv')
    X_test = pd.read_csv(model_training_dir / 'X_test.csv')
    y_train = pd.read_csv(model_training_dir / 'y_train.csv')
    y_val = pd.read_csv(model_training_dir / 'y_val.csv')
    y_test = pd.read_csv(model_training_dir / 'y_test.csv')
    with open(model_training_dir / 'columns_schema.json', 'r') as f:
        schema = json.load(f)

    # Schema consistency checks
    assert schema['model_input_cols'] == list(X_train.columns)
    assert schema['y_cols'] == list(y_train.columns)
    assert len(schema['y_cont_cols']) + len(schema['y_bin_cols']) == len(schema['y_cols'])
    assert schema['num_binary_outputs'] == len(schema['y_bin_cols'])
    assert schema['num_continuous_outputs'] == len(schema['y_cont_cols'])

    # X_val and y_val should retain subject_id and align to train columns
    assert list(X_val.columns) == ['subject_id'] + schema['model_input_cols']
    assert list(y_val.columns) == ['subject_id'] + schema['y_cols']

    # y_* should not contain action columns
    assert not any(c.endswith('_active') for c in y_train.columns)
    assert not any(c.endswith('_active') for c in y_test.columns)
    assert not any(c.endswith('_active') for c in y_val.columns)

    # Action cols recorded in schema and present in X_* only
    assert all(c.endswith('_active') for c in schema['action_cols'])
    for c in schema['action_cols']:
        assert c in X_train.columns and c not in y_train.columns

    # Scalers are validated separately in dedicated tests; schema and artifacts suffice here.


def test_scalers_round_trip_and_expected_columns(tmp_path, monkeypatch):
    """
    Validates that:
    - normalize_and_encode saves scaler_X and scaler_y with expected feature columns
      (X: CONTINUOUS_VARS + ORDINAL_VARS + ['time_delta']; y: CONTINUOUS_VARS + ORDINAL_VARS)
    - Both scalers can round-trip (inverse_transform(transform(data)) ~= data)
    - Expected columns presence matches what rollout_evaluation.py relies on.
    """
    import joblib
    import utils as utils_mod

    # Point MODEL_TRAINING_DIR to temp for saving scalers
    monkeypatch.setattr(utils_mod, 'MODEL_TRAINING_DIR', str(tmp_path), raising=False)

    # Define which variables get scaled
    monkeypatch.setattr(utils_mod, 'CONTINUOUS_VARS', ['continuous_feature'], raising=False)
    monkeypatch.setattr(utils_mod, 'ORDINAL_VARS', ['ordinal_feature'], raising=False)

    # Build tiny splits; include categorical columns and an action-like column
    X_train = pd.DataFrame({
        'continuous_feature': [0.0, 2.0, -1.0],
        'ordinal_feature': [1.0, 3.0, 5.0],
        'time_delta': [1.0, 2.0, 3.0],
        'PTGENDER': ['Male', 'Female', 'Male'],
        'PTRACCAT': ['White', 'Black', 'White'],
        'A_active': [0, 1, 0],
    })
    X_val = pd.DataFrame({
        'subject_id': [10, 11],
        'continuous_feature': [1.0, 1.0],
        'ordinal_feature': [2.0, 2.0],
        'time_delta': [1.5, 1.5],
        'PTGENDER': ['Male', 'Male'],
        'PTRACCAT': ['White', 'White'],
        'A_active': [1, 0],
    })
    X_test = pd.DataFrame({
        'continuous_feature': [4.0],
        'ordinal_feature': [5.0],
        'time_delta': [3.0],
        'PTGENDER': ['Female'],
        'PTRACCAT': ['Black'],
        'A_active': [0],
    })

    y_train = pd.DataFrame({
        'continuous_feature': [10.0, 20.0, 30.0],
        'ordinal_feature': [1.0, 2.0, 3.0],
    })
    y_val = pd.DataFrame({
        'subject_id': [10, 11],
        'continuous_feature': [15.0, 25.0],
        'ordinal_feature': [1.5, 2.5],
    })
    y_test = pd.DataFrame({
        'continuous_feature': [35.0],
        'ordinal_feature': [3.5],
    })

    # Run normalization/encoding to produce scalers
    processed_X_train, processed_X_val, processed_X_test, processed_y_train, processed_y_val, processed_y_test = utils_mod.normalize_and_encode(
        X_train.copy(), X_val.copy(), X_test.copy(),
        y_train.copy(), y_val.copy(), y_test.copy()
    )

    # Scalers saved (by normalize_and_encode) to MODEL_TRAINING_DIR as 'scaler_X.joblib'/'scaler_y.joblib'
    scaler_X_path = tmp_path / 'scaler_X.joblib'
    scaler_y_path = tmp_path / 'scaler_y.joblib'
    assert scaler_X_path.exists() and scaler_y_path.exists()

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Expected columns in scalers
    expected_X_cols = ['continuous_feature', 'ordinal_feature', 'time_delta']
    expected_y_cols = ['continuous_feature', 'ordinal_feature']

    assert hasattr(scaler_X, 'feature_names_in_')
    assert hasattr(scaler_y, 'feature_names_in_')
    assert list(scaler_X.feature_names_in_) == expected_X_cols
    assert list(scaler_y.feature_names_in_) == expected_y_cols

    # Round-trip test for X scaler
    original_X_numeric = X_train[expected_X_cols]
    scaled_X = scaler_X.transform(original_X_numeric)
    unscaled_X = scaler_X.inverse_transform(scaled_X)
    np.testing.assert_allclose(unscaled_X, original_X_numeric.values, rtol=1e-6, atol=1e-6)

    # Round-trip test for y scaler
    original_y_numeric = y_train[expected_y_cols]
    scaled_y = scaler_y.transform(original_y_numeric)
    unscaled_y = scaler_y.inverse_transform(scaled_y)
    np.testing.assert_allclose(unscaled_y, original_y_numeric.values, rtol=1e-6, atol=1e-6)

    # Simulate rollout unscaling checks: verify required columns are present
    # Build a predicted trajectory frame containing both target and action features
    pred_df = pd.DataFrame({
        'continuous_feature': scaled_y[:, 0],
        'ordinal_feature': scaled_y[:, 1],
        'A_active': X_train['A_active'].iloc[: scaled_y.shape[0]].values,
        'time_delta': X_train['time_delta'].iloc[: scaled_y.shape[0]].values,
    })
    # Ensure scaler_y columns exist and can be inverse transformed
    pred_y_block = pred_df[expected_y_cols]
    _ = scaler_y.inverse_transform(pred_y_block)
    # Ensure scaler_X columns exist and can be inverse transformed
    pred_x_block = pred_df[expected_X_cols]
    _ = scaler_X.inverse_transform(pred_x_block)

    # Columns that should not be part of any scaler
    assert 'A_active' not in scaler_X.feature_names_in_
    assert 'A_active' not in scaler_y.feature_names_in_
    # Categorical one-hot columns are created after scaling and are not part of scaler_y
    assert 'PTRACCAT_White' not in scaler_y.feature_names_in_
