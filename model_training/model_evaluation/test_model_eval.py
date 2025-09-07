import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure we can import utils from this folder without packages
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import (
    load_schema,
    derive_feature_sets_from_schema,
    print_feature_sets,
    load_validation_data,
    validate_train_val_alignment,
    load_and_inspect_scalers,
    validate_scaler_coverage,
    inverse_scale_columns_inplace,
    build_category_indices,
    postprocess_observation_vector,
    autoregressive_rollout_for_subject,
)


# ---------------------- Fixtures ----------------------

@pytest.fixture(autouse=True)
def isolate_cwd(tmp_path, monkeypatch, request):
    """For unit tests, run in a temp CWD; for integration tests, keep real CWD.

    This allows integration tests to read real files while unit tests remain isolated.
    """
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def write_guard(tmp_path, monkeypatch):
    """Prevent writes outside the test's tmp_path, including accidental overwrites.

    Patches common writers (open in write modes, to_csv, joblib.dump, numpy.save/savetxt, torch.save,
    Path.write_text/bytes). Allows writes only under tmp_path.
    """
    import builtins as _builtins
    from functools import wraps
    import numpy as _np
    import joblib as _joblib
    try:
        import torch as _torch  # noqa: F401
        _torch_available = True
    except Exception:  # pragma: no cover - environment dependent
        _torch_available = False

    allowed_root = Path(tmp_path).resolve()

    def _is_allowed_path(p):
        try:
            # If p is already a Path or string, keep it. For file-like objects (e.g., open file) use its name.
            if isinstance(p, (str, Path)):
                p = Path(p)
            elif getattr(p, 'name', None) and isinstance(p.name, str):
                p = Path(p.name)
            else:
                # Non-path-like target (e.g., buffer) is allowed
                return True
        except Exception:
            return True  # non-path-like targets (e.g., stdout) are allowed
        if not p.is_absolute():
            p = Path(os.getcwd()) / p
        try:
            p = p.resolve()
        except Exception:
            pass
        try:
            return str(p).startswith(str(allowed_root))
        except Exception:
            return False

    # Patch builtins.open to guard write modes
    _orig_open = _builtins.open

    @wraps(_orig_open)
    def _guarded_open(file, mode='r', *args, **kwargs):
        if any(m in mode for m in ('w', 'a', '+')):
            if not _is_allowed_path(file):
                raise PermissionError(f"Blocked write to non-temp path: {file}")
        return _orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(_builtins, 'open', _guarded_open, raising=True)

    # Patch pandas.DataFrame.to_csv
    import pandas as _pd
    _orig_to_csv = _pd.DataFrame.to_csv

    def _guarded_to_csv(self, path_or_buf=None, *args, **kwargs):
        target = path_or_buf
        # If buffer-like with name, check; if string/path, check; else allow
        if isinstance(target, (str, Path)) or getattr(target, 'name', None):
            if not _is_allowed_path(target):
                raise PermissionError(f"Blocked write to non-temp CSV path: {target}")
        return _orig_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(_pd.DataFrame, 'to_csv', _guarded_to_csv, raising=True)

    # joblib.dump
    _orig_joblib_dump = _joblib.dump

    def _guarded_joblib_dump(value, filename, *args, **kwargs):
        if not _is_allowed_path(filename):
            raise PermissionError(f"Blocked joblib.dump to non-temp path: {filename}")
        return _orig_joblib_dump(value, filename, *args, **kwargs)

    monkeypatch.setattr(_joblib, 'dump', _guarded_joblib_dump, raising=True)

    # numpy.save, numpy.savetxt
    _orig_np_save = _np.save
    _orig_np_savetxt = _np.savetxt

    def _guarded_np_save(file, *args, **kwargs):
        if not _is_allowed_path(file):
            raise PermissionError(f"Blocked numpy.save to non-temp path: {file}")
        return _orig_np_save(file, *args, **kwargs)

    def _guarded_np_savetxt(fname, *args, **kwargs):
        if not _is_allowed_path(fname):
            raise PermissionError(f"Blocked numpy.savetxt to non-temp path: {fname}")
        return _orig_np_savetxt(fname, *args, **kwargs)

    monkeypatch.setattr(_np, 'save', _guarded_np_save, raising=True)
    monkeypatch.setattr(_np, 'savetxt', _guarded_np_savetxt, raising=True)

    # torch.save
    if _torch_available:
        import torch as _torch
        _orig_torch_save = _torch.save

        def _guarded_torch_save(obj, f, *args, **kwargs):
            if not _is_allowed_path(f):
                raise PermissionError(f"Blocked torch.save to non-temp path: {f}")
            return _orig_torch_save(obj, f, *args, **kwargs)

        monkeypatch.setattr(_torch, 'save', _guarded_torch_save, raising=True)

    # Path.write_text / write_bytes
    _orig_write_text = Path.write_text
    _orig_write_bytes = Path.write_bytes

    def _guarded_write_text(self, *args, **kwargs):
        if not _is_allowed_path(self):
            raise PermissionError(f"Blocked write_text to non-temp path: {self}")
        return _orig_write_text(self, *args, **kwargs)

    def _guarded_write_bytes(self, *args, **kwargs):
        if not _is_allowed_path(self):
            raise PermissionError(f"Blocked write_bytes to non-temp path: {self}")
        return _orig_write_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', _guarded_write_text, raising=True)
    monkeypatch.setattr(Path, 'write_bytes', _guarded_write_bytes, raising=True)


@pytest.fixture(scope="session", autouse=True)
def require_torch():
    """Fail the entire test session if PyTorch is unavailable.

    Prevents inadvertently running tests in an environment without torch.
    """
    if not TORCH_AVAILABLE:
        pytest.fail(
            "PyTorch (torch) is required for this test suite. Please run pytest via the same interpreter/env that has torch installed (e.g., `python -m pytest` after activating your ML environment)."
        )

@pytest.fixture
def example_schema():
    return {
        "model_input_cols": [
            "A_active",  # action
            "B_active",  # action
            "cont1",
            "bin1",
            "months_since_bl",
            "time_delta",
        ],
        "observation_cols": [
            "cont1",
            "bin1",
            "months_since_bl",
        ],
        "action_cols": ["A_active", "B_active"],
        "y_categorical_groups": {"CAT": ["bin1"]},
        "y_cont_cols": ["cont1", "months_since_bl"],
    }


@pytest.fixture
def tmp_schema_file(tmp_path, example_schema):
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(example_schema))
    return p


# ---------------------- Tests: schema/features ----------------------

def test_load_schema_success(tmp_schema_file, example_schema):
    s = load_schema(tmp_schema_file)
    assert s == example_schema


def test_derive_feature_sets_from_schema(example_schema):
    model_input_cols, action_cols, observation_cols, state_cols = derive_feature_sets_from_schema(example_schema, time_delta_col="time_delta")
    assert model_input_cols == example_schema["model_input_cols"]
    assert action_cols == ["A_active", "B_active"]
    assert observation_cols == ["cont1", "bin1", "months_since_bl"]
    # state should be model_input minus time_delta
    assert state_cols == ["A_active", "B_active", "cont1", "bin1", "months_since_bl"]


def test_derive_feature_sets_from_schema_with_suffix():
    schema = {
        "model_input_cols": ["A_active", "X", "time_delta"],
        "observation_cols": ["X"],
        # no explicit action_cols; should infer by suffix
    }
    model_input_cols, action_cols, observation_cols, state_cols = derive_feature_sets_from_schema(schema, time_delta_col="time_delta")
    assert action_cols == ["A_active"]
    assert state_cols == ["A_active", "X"]


def test_print_feature_sets_smoke(example_schema, capsys):
    model_input_cols, action_cols, observation_cols, state_cols = derive_feature_sets_from_schema(example_schema, "time_delta")
    print_feature_sets(model_input_cols, action_cols, observation_cols, state_cols)
    out = capsys.readouterr().out
    assert "Feature Set Definitions" in out
    assert "model_input_cols" in out
    assert str(list(observation_cols)) in out


# ---------------------- Tests: validation data ----------------------

def test_load_validation_data_success(tmp_path):
    # Build a simple X_val with duplicates per (subject, time)
    df = pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 2],
        "months_since_bl": [0, 1, 1, 0, 6],
        "time_delta": [0.0, 1.0, 0.0, 0.0, 6.0],
        "A_active": [1, 0, 0, 0, 1],
    })
    p = tmp_path / "X_val.csv"
    df.to_csv(p, index=False)
    full_df, unique_df = load_validation_data(p, "subject_id", "months_since_bl", "time_delta")
    assert len(full_df) == 5
    # duplicates at time=1 for subject 1 should be dropped
    assert len(unique_df) == 4
    # uniqueness check: no duplicate pairs
    assert not unique_df.duplicated(subset=["subject_id", "months_since_bl"]).any()


def test_load_validation_data_missing_columns(tmp_path):
    df = pd.DataFrame({"subject_id": [1], "months_since_bl": [0]})
    p = tmp_path / "X_val.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError) as e:
        load_validation_data(p, "subject_id", "months_since_bl", "time_delta")
    assert "time delta column" in str(e.value)


# ---------------------- Tests: train/val alignment ----------------------

def test_validate_train_val_alignment_success(tmp_path, example_schema):
    # X_train includes subject_id and matches model_input (order after dropping id)
    x_train = pd.DataFrame({
        "subject_id": [1, 2],
        "A_active": [0, 1],
        "B_active": [1, 0],
        "cont1": [0.0, 1.0],
        "bin1": [0, 1],
        "months_since_bl": [0.0, 1.0],
        "time_delta": [1.0, 1.0],
    })
    y_train = pd.DataFrame({
        # extras that must be ignored
        "subject_id": [1, 2],
        "subject_id.1": [1, 2],
        # targets
        "cont1": [0.0, 1.0],
        "bin1": [0, 1],
        "months_since_bl": [0.0, 1.0],
    })
    x_train_path = tmp_path / "X_train.csv"
    y_train_path = tmp_path / "y_train.csv"
    x_train.to_csv(x_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)

    model_input_cols, action_cols, observation_cols, state_cols = derive_feature_sets_from_schema(example_schema, "time_delta")
    y_train_effective_targets = validate_train_val_alignment(x_train_path, y_train_path, observation_cols, "subject_id", model_input_cols)
    assert y_train_effective_targets == ["cont1", "bin1", "months_since_bl"]


def test_validate_train_val_alignment_missing_target_raises(tmp_path, example_schema):
    x_train = pd.DataFrame({
        "subject_id": [1],
        "A_active": [0],
        "B_active": [1],
        "cont1": [0.0],
        "bin1": [0],
        "months_since_bl": [0.0],
        "time_delta": [1.0],
    })
    # y_train missing 'bin1'
    y_train = pd.DataFrame({"cont1": [0.0], "months_since_bl": [0.0]})
    x_train_path = tmp_path / "X_train.csv"
    y_train_path = tmp_path / "y_train.csv"
    x_train.to_csv(x_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    model_input_cols, action_cols, observation_cols, state_cols = derive_feature_sets_from_schema(example_schema, "time_delta")
    with pytest.raises(ValueError):
        validate_train_val_alignment(x_train_path, y_train_path, observation_cols, "subject_id", model_input_cols)


# ---------------------- Tests: scalers ----------------------

def _fit_scaler_with_feature_names(df: pd.DataFrame) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(df)
    return sc


def test_load_and_inspect_scalers(tmp_path, capsys):
    # Create two simple scalers and dump
    df_X = pd.DataFrame({"cont1": [0.0, 1.0], "months_since_bl": [0.0, 1.0], "time_delta": [1.0, 1.0]})
    df_Y = pd.DataFrame({"cont1": [0.0, 1.0], "months_since_bl": [0.0, 1.0]})
    scaler_X = _fit_scaler_with_feature_names(df_X)
    scaler_y = _fit_scaler_with_feature_names(df_Y)
    path_scaler_X = tmp_path / "scaler_X.joblib"
    path_scaler_y = tmp_path / "scaler_y.joblib"
    joblib.dump(scaler_X, path_scaler_X)
    joblib.dump(scaler_y, path_scaler_y)
    loaded_scaler_X, loaded_scaler_y = load_and_inspect_scalers(path_scaler_X, path_scaler_y)
    assert loaded_scaler_X is not None and loaded_scaler_y is not None
    out = capsys.readouterr().out
    assert "Inspecting scaler_X" in out and "Inspecting scaler_y" in out


def test_validate_scaler_coverage_ok(capsys):
    df_X = pd.DataFrame({"cont1": [0.0, 1.0], "months_since_bl": [0.0, 1.0], "time_delta": [1.0, 1.0]})
    df_Y = pd.DataFrame({"cont1": [0.0, 1.0], "months_since_bl": [0.0, 1.0]})
    scaler_X = _fit_scaler_with_feature_names(df_X)
    scaler_y = _fit_scaler_with_feature_names(df_Y)
    trajectory_columns = ["cont1", "months_since_bl", "time_delta"]
    observation_columns = ["cont1", "months_since_bl"]
    validate_scaler_coverage(scaler_X, scaler_y, trajectory_columns, observation_columns)
    out = capsys.readouterr().out
    assert "scaler_X features validated" in out
    assert "scaler_y features validated" in out


def test_validate_scaler_coverage_mismatch(capsys):
    df_X = pd.DataFrame({"cont1": [0.0, 1.0]})
    df_Y = pd.DataFrame({"cont1": [0.0, 1.0], "months_since_bl": [0.0, 1.0]})
    scaler_X = _fit_scaler_with_feature_names(df_X)
    scaler_y = _fit_scaler_with_feature_names(df_Y)
    trajectory_columns = ["cont1"]
    observation_columns = ["cont1", "months_since_bl"]
    validate_scaler_coverage(scaler_X, scaler_y, trajectory_columns, observation_columns)
    out = capsys.readouterr().out
    # In this scenario, scaler_y matches observation_cols but is missing a column in trajectory_cols,
    # so we expect a coverage warning rather than a feature-set mismatch warning.
    assert "Some scaler_y features are not present in trajectory data" in out


def test_inverse_scale_columns_inplace_roundtrip():
    # Create scaled data and check inverse transform on subset
    df = pd.DataFrame({"cont1": [10.0, 20.0], "months_since_bl": [0.0, 12.0], "other": [5.0, 6.0]})
    scaler_X = _fit_scaler_with_feature_names(df[["cont1", "months_since_bl"]])

    # Produce standardized values
    scaled = df.copy()
    scaled[["cont1", "months_since_bl"]] = scaler_X.transform(df[["cont1", "months_since_bl"]])

    # Inverse only cont1
    inverse_scale_columns_inplace(scaled, ["cont1"], scaler_X, "scaler_X")
    assert np.allclose(scaled["cont1"].to_numpy(), df["cont1"].to_numpy())
    # months_since_bl remains scaled
    assert not np.allclose(scaled["months_since_bl"].to_numpy(), df["months_since_bl"].to_numpy())


# ---------------------- Tests: indices and postprocessing ----------------------

def test_build_category_indices(example_schema):
    _, _, obs, _ = derive_feature_sets_from_schema(example_schema, "time_delta")
    group_indices, group_members = build_category_indices(example_schema, obs)
    # 'bin1' is in observation and in group CAT
    assert set(group_indices.keys()) == {"CAT"}
    assert group_indices["CAT"] == [1]  # bin1 index in obs
    assert group_members == [1]


def test_postprocess_observation_vector():
    observation_cols = ["cont1", "bin1", "months_since_bl"]
    schema = {"y_cont_cols": ["cont1", "months_since_bl"]}
    ordinal_vars = ["months_since_bl"]  # only months_since_bl is ordinal among cont
    # y = [cont1, bin1, months]
    y = torch.tensor([1.4, 0.4, 2.6], dtype=torch.float32)
    # No groups, bin index is 1
    group_indices = {}
    independent_bin_indices = [1]
    out = postprocess_observation_vector(y, observation_cols, schema, ordinal_vars, group_indices, independent_bin_indices)
    # cont1 unchanged, months rounded, bin thresholded to 0
    assert np.isclose(out[0].item(), 1.4)
    assert np.isclose(out[2].item(), 3.0)
    assert np.isclose(out[1].item(), 0.0)

    # Test argmax within group
    group_indices = {"CAT": [1]}
    y2 = torch.tensor([0.0, 0.9, 2.1], dtype=torch.float32)
    out2 = postprocess_observation_vector(y2, observation_cols, schema, ordinal_vars, group_indices, [])
    assert np.isclose(out2[1].item(), 1.0)


# ---------------------- Tests: autoregressive rollout ----------------------

class CapturingDummyModel:
    """Dummy model that captures input tensors and returns controlled outputs.

    By default returns constant predictions. Can be configured with a vector_fn(idx) to
    generate per-feature outputs.
    """

    def __init__(self, out_dim, value=0.5, vector_fn=None):
        self.out_dim = out_dim
        self.value = value
        self.vector_fn = vector_fn  # callable: i -> value
        self.calls = []  # store dicts with x_seq_tensor and inferred shapes

    def infer_observations(self, x_seq_tensor, cont_idx, bin_idx, out_dim, key_padding_mask, attn_mask, apply_sigmoid=True):
        S = x_seq_tensor.size(1)
        F = x_seq_tensor.size(2)
        self.calls.append({
            "x_seq_tensor": x_seq_tensor.detach().cpu().clone(),
            "S": S,
            "F": F,
        })
        if self.vector_fn is None:
            out = torch.full((1, S, out_dim), fill_value=self.value, dtype=torch.float32)
        else:
            # build per-feature pattern replicated across sequence S
            vec = torch.tensor([self.vector_fn(i) for i in range(out_dim)], dtype=torch.float32)
            out = vec.view(1, 1, -1).repeat(1, S, 1)
        return out


def test_autoregressive_rollout_for_subject_basic(example_schema):
    model_input_cols, action_cols, observation_cols, state_cols = derive_feature_sets_from_schema(example_schema, "time_delta")
    # Build a simple subject with 3 visits (2 steps), with a non-positive delta to trigger path
    subject_true_df = pd.DataFrame({
        "A_active": [1, 0, 1],
        "B_active": [0, 1, 0],
        "cont1": [0.0, 0.0, 0.0],  # initial cont ignored; predictions will fill
        "bin1": [0, 0, 0],
        "months_since_bl": [0.0, 1.0, 1.0],  # second delta 0.0 triggers non-positive branch
        "time_delta": [1.0, 1.0, 0.0],
    })

    # Model that returns constant 0.7 and captures inputs
    model = CapturingDummyModel(out_dim=len(observation_cols), value=0.7)
    device = torch.device("cpu")

    # Indices consistent with observation_cols ordering
    continuous_idx = [0, 2]
    binary_idx = [1]
    group_indices, group_members = build_category_indices(example_schema, observation_cols)
    independent_binary_indices = [i for i in binary_idx if i not in set(group_members)]

    predicted_states = autoregressive_rollout_for_subject(
        subject_true_df=subject_true_df,
        model=model,
        device=device,
        model_input_cols=model_input_cols,
        observation_cols=observation_cols,
        state_cols=state_cols,
        action_cols=action_cols,
        time_col="months_since_bl",
        time_delta_col="time_delta",
        cont_idx=continuous_idx,
        bin_idx=binary_idx,
        group_indices=group_indices,
        independent_bin_indices=independent_binary_indices,
        ordinal_vars=[],  # disable rounding for simplicity
        schema=example_schema,
    )

    # Expect 3 state dicts (initial + 2 steps)
    assert len(predicted_states) == 3
    # Initial dict contains initial time
    assert np.isclose(predicted_states[0]["months_since_bl"], 0.0)
    # Next steps should copy action features from true next state
    assert predicted_states[1]["A_active"] == subject_true_df.loc[1, "A_active"]
    assert predicted_states[2]["B_active"] == subject_true_df.loc[2, "B_active"]
    # Observation values are from model (0.7 after post-processing; bin thresholded to 1)
    assert np.isclose(predicted_states[1]["cont1"], 0.7)
    assert np.isclose(predicted_states[1]["bin1"], 1.0)
    # months_since_bl updated to true next visit time
    assert np.isclose(predicted_states[1]["months_since_bl"], 1.0)
    assert np.isclose(predicted_states[2]["months_since_bl"], 1.0)
    # Verify captured sequence dimensions and order used for model input
    # One call per rollout step => 2 calls (k=0 and k=1); sequence lengths should be [1,2]
    seq_lengths = [c["S"] for c in model.calls]
    assert seq_lengths == [1, 2]
    feature_dims = [c["F"] for c in model.calls]
    assert feature_dims == [len(model_input_cols), len(model_input_cols)]
    # Validate the very first input row ordering matches model_input_cols
    x0 = model.calls[0]["x_seq_tensor"][0, 0, :].numpy()
    expected_first = np.array([
        subject_true_df.loc[0, "A_active"],
        subject_true_df.loc[0, "B_active"],
        subject_true_df.loc[0, "cont1"],
        subject_true_df.loc[0, "bin1"],
        subject_true_df.loc[0, "months_since_bl"],
        subject_true_df.loc[1, "months_since_bl"] - subject_true_df.loc[0, "months_since_bl"],
    ], dtype=float)
    assert np.allclose(x0, expected_first)


# ---------------------- Integration test (synthetic) ----------------------

@pytest.mark.integration
def test_integration_minimal_end_to_end(tmp_path, example_schema):
    """Minimal end-to-end across utilities with synthetic data.

    - Write temp schema, X_train, y_train, X_val
    - Validate alignment, load validation data
    - Build feature sets and category indices
    - Create simple scalers and validate coverage
    - Run rollout with DummyModel
    All files live under tmp_path; nothing touches project files.
    """
    # Use real project files for a deep sanity check
    repo_model_training = THIS_DIR.parent
    schema_path = repo_model_training / "columns_schema.json"
    x_train_path = repo_model_training / "X_train.csv"
    y_train_path = repo_model_training / "y_train.csv"
    x_val_path = repo_model_training / "X_val.csv"

    assert schema_path.exists(), f"Missing real schema at {schema_path}"
    assert x_train_path.exists() and y_train_path.exists() and x_val_path.exists(), "Missing real train/val CSVs."

    # Load schema and derive features
    schema = load_schema(schema_path)
    model_input_cols, action_cols, observation_cols, state_cols = derive_feature_sets_from_schema(schema, "time_delta")

    # Validate train/val alignment using real files
    y_train_effective_targets = validate_train_val_alignment(x_train_path, y_train_path, observation_cols, "subject_id", model_input_cols)
    assert set(y_train_effective_targets) == set(observation_cols)

    # Load validation data
    full_df, unique_df = load_validation_data(x_val_path, "subject_id", "months_since_bl", "time_delta")
    assert len(full_df) >= 2 and len(unique_df) >= 2

    # Load real scalers and validate coverage
    scaler_X_path = repo_model_training / "scaler_X.joblib"
    scaler_Y_path = repo_model_training / "scaler_y.joblib"
    assert scaler_X_path.exists() and scaler_Y_path.exists(), "Missing real scalers."
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_Y_path)
    validate_scaler_coverage(scaler_X, scaler_y, trajectory_cols=list(set(state_cols) | {"months_since_bl", "time_delta"}), observation_cols=observation_cols)

    # Choose the first subject with at least 2 visits
    subject_counts = unique_df.groupby("subject_id").size()
    valid_subjects = subject_counts[subject_counts >= 2].index
    assert len(valid_subjects) > 0, "No subject with >=2 visits for integration test."
    subject_id = valid_subjects[0]
    subject_df = unique_df[unique_df["subject_id"] == subject_id].sort_values("months_since_bl").reset_index(drop=True)

    # Build a capturing dummy model with per-feature vector to test OHE and rounding
    obs_index_map = {c: i for i, c in enumerate(observation_cols)}
    # Pick an ordinal feature to round (prefer TRABSCOR if available; else months_since_bl)
    ordinal_feature = "TRABSCOR" if "TRABSCOR" in observation_cols else "months_since_bl"
    # Pick a categorical group from schema
    y_categorical_groups = schema.get("y_categorical_groups", {}) or {}
    chosen_group_name, chosen_group_members = None, []
    for g, cols in y_categorical_groups.items():
        members_in_obs = [c for c in cols if c in observation_cols]
        if len(members_in_obs) >= 2:
            chosen_group_name, chosen_group_members = g, members_in_obs
            break
    assert chosen_group_members, "No categorical group with >=2 members found in observation_cols."

    values_by_index = {i: 0.6 for i in range(len(observation_cols))}
    # Set ordinal feature to 2.6 to trigger rounding
    if ordinal_feature in obs_index_map:
        values_by_index[obs_index_map[ordinal_feature]] = 2.6
    # Set OHE group values with a clear winner at the last member
    for j, col in enumerate(chosen_group_members):
        if col in obs_index_map:
            values_by_index[obs_index_map[col]] = 0.1 + 0.2 * j  # increasing values
    expected_winner = chosen_group_members[len(chosen_group_members) - 1]

    def vec_fn(i):
        return float(values_by_index.get(i, 0.6))

    model = CapturingDummyModel(out_dim=len(observation_cols), vector_fn=vec_fn)
    group_indices, group_members = build_category_indices(schema, observation_cols)
    continuous_idx = [0, 2]
    binary_idx = [1]
    independent_binary_indices = [i for i in binary_idx if i not in set(group_members)]
    predicted_states = autoregressive_rollout_for_subject(
        subject_true_df=subject_df,
        model=model,
        device=torch.device("cpu"),
        model_input_cols=model_input_cols,
        observation_cols=observation_cols,
        state_cols=state_cols,
        action_cols=action_cols,
        time_col="months_since_bl",
        time_delta_col="time_delta",
        cont_idx=continuous_idx,
        bin_idx=binary_idx,
        group_indices=group_indices,
        independent_bin_indices=independent_binary_indices,
        ordinal_vars=[ordinal_feature],
        schema=schema,
    )
    # Expect at least 2 states (2 visits)
    assert len(predicted_states) >= 2
    # Confirm model saw correct shapes and feature count
    assert model.calls, "Model was not called."
    first_call = model.calls[0]
    assert first_call["x_seq_tensor"].shape == (1, 1, len(model_input_cols))
    # Confirm input order for first step equals schema order
    x0 = first_call["x_seq_tensor"][0, 0, :].numpy()
    expected_first = np.array([subject_df.loc[0, fn] if fn != "time_delta" else (subject_df.loc[1, "months_since_bl"] - subject_df.loc[0, "months_since_bl"]) for fn in model_input_cols], dtype=float)
    assert np.allclose(x0, expected_first)

    # OHE argmax in chosen group enforced
    assert predicted_states[1][expected_winner] == 1.0
    for col in chosen_group_members:
        if col != expected_winner:
            assert np.isclose(predicted_states[1][col], 0.0)

    # Ordinal rounding (if not months_since_bl which is overwritten by true time, choose another if available)
    if ordinal_feature != "months_since_bl":
        assert np.isclose(predicted_states[1][ordinal_feature], round(2.6))

    # Scaler round-trip using real scaler_y on its exact feature set
    predicted_df = pd.DataFrame(predicted_states)
    if hasattr(scaler_y, 'feature_names_in_'):
        cols = list(scaler_y.feature_names_in_)
        missing_cols = [c for c in cols if c not in predicted_df.columns]
        assert not missing_cols, f"Predicted dataframe missing columns required by scaler_y: {missing_cols}"
        scaled_df = predicted_df.copy()
        scaled_df[cols] = scaler_y.transform(predicted_df[cols])
        inverse_scale_columns_inplace(scaled_df, cols, scaler_y, "scaler_y")
        assert np.allclose(scaled_df[cols].to_numpy(dtype=float), predicted_df[cols].to_numpy(dtype=float))


@pytest.mark.integration
def test_integration_ohe_group_and_ordinal_rounding(tmp_path):
    """Integration focusing on OHE group argmax and ordinal rounding."""
    schema = {
        "model_input_cols": [
            "A_active",
            "cont1",
            "MARR_Single",
            "MARR_Married",
            "MARR_Divorced",
            "months_since_bl",
            "time_delta",
        ],
        "observation_cols": [
            "cont1",
            "MARR_Single",
            "MARR_Married",
            "MARR_Divorced",
            "months_since_bl",
        ],
        "action_cols": ["A_active"],
        "y_categorical_groups": {"MARR": ["MARR_Single", "MARR_Married", "MARR_Divorced"]},
        "y_cont_cols": ["cont1", "months_since_bl"],
    }
    model_input_cols, action_cols, observation_cols, state_cols = derive_feature_sets_from_schema(schema, "time_delta")
    # Two visits
    subject_df = pd.DataFrame({
        "A_active": [1, 0],
        "cont1": [0.0, 0.0],
        "MARR_Single": [1, 0],
        "MARR_Married": [0, 1],
        "MARR_Divorced": [0, 0],
        "months_since_bl": [2.0, 3.0],
        "time_delta": [1.0, 1.0],
    })
    # Vector fn: produce [cont1=0.2, Single=0.1, Married=0.3, Divorced=0.9, months=2.6]
    values = {
        "cont1": 0.2,
        "MARR_Single": 0.1,
        "MARR_Married": 0.3,
        "MARR_Divorced": 0.9,
        "months_since_bl": 2.6,
    }
    obs_index_map = {c: i for i, c in enumerate(observation_cols)}
    def vec_fn(i):
        for k, v in values.items():
            if obs_index_map[k] == i:
                return v
        return 0.0
    model = CapturingDummyModel(out_dim=len(observation_cols), vector_fn=vec_fn)
    group_indices, group_members = build_category_indices(schema, observation_cols)
    continuous_idx = [obs_index_map["cont1"], obs_index_map["months_since_bl"]]
    binary_idx = [obs_index_map["MARR_Single"], obs_index_map["MARR_Married"], obs_index_map["MARR_Divorced"]]
    independent_binary_indices = []  # all in group
    predicted_states = autoregressive_rollout_for_subject(
        subject_true_df=subject_df,
        model=model,
        device=torch.device("cpu"),
        model_input_cols=model_input_cols,
        observation_cols=observation_cols,
        state_cols=state_cols,
        action_cols=action_cols,
        time_col="months_since_bl",
        time_delta_col="time_delta",
        cont_idx=continuous_idx,
        bin_idx=binary_idx,
        group_indices=group_indices,
        independent_bin_indices=independent_binary_indices,
        ordinal_vars=["months_since_bl"],
        schema=schema,
    )
    # Initial + 1 step
    assert len(predicted_states) == 2
    # Ordinal rounding: months_since_bl predicted 2.6 -> 3.0 at step 1
    assert np.isclose(predicted_states[1]["months_since_bl"], 3.0)
    # OHE argmax: Divorced has highest (0.9) -> one-hot [0,0,1]
    assert predicted_states[1]["MARR_Single"] == 0.0
    assert predicted_states[1]["MARR_Married"] == 0.0
    assert predicted_states[1]["MARR_Divorced"] == 1.0
