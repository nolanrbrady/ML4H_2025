import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from .alpaca_env import ALPACAEnv


@pytest.fixture(scope="module")
def alpaca_dir() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope="module")
def env(alpaca_dir):
    # Use the local ALPACA artifacts (scalers, model, schema, bounds)
    return ALPACAEnv(data_path=str(alpaca_dir))


def test_schema_and_model_inputs_match_scalers(env: ALPACAEnv):
    # model_input_cols should include obs + actions + next_visit_months (order defined by schema)
    assert isinstance(env.model_input_cols, list) and len(env.model_input_cols) > 0

    # Scaler-X columns must be a subset of model_input_cols and transform should be consistent
    sx_cols = list(getattr(env.scaler_X, 'feature_names_in_', []))
    assert set(sx_cols).issubset(set(env.model_input_cols))

    # Build a model input row like env.step constructs, test round-trip scaling on intersecting cols
    obs, _ = env.reset()
    obs_s = pd.Series(obs, index=env.observation_cols)
    act = np.zeros(len(env.action_cols), dtype=float)
    act_s = pd.Series(act, index=env.action_cols)
    row = pd.concat([obs_s, act_s])
    row[env.delta_col] = env.time_delta_val

    df = pd.DataFrame([row])[env.model_input_cols]
    cols = [c for c in sx_cols if c in df.columns]
    if cols:
        before = df[cols].copy()
        scaled = env.manage_state_scaling(df[cols], env.scaler_X, normalize=True)
        unscaled = env.manage_state_scaling(scaled, env.scaler_X, normalize=False)
        # Round-trip should match original within numerical tolerances
        np.testing.assert_allclose(unscaled.values, before.values, rtol=1e-5, atol=1e-5)


def test_spaces_and_bounds(env: ALPACAEnv):
    # Action/observation shape checks
    assert env.action_space.n == len(env.action_cols)
    assert env.observation_space.shape[0] == len(env.observation_cols)
    assert np.all(env.observation_space.low < env.observation_space.high)


def test_action_constraints(env: ALPACAEnv):
    env.reset()
    # No action -> done with strong negative reward
    no_action = np.zeros(len(env.action_cols), dtype=int)
    _, reward, done, _, info = env.step(no_action)
    assert done and reward == -10.0

    # No Medication concurrent with others -> done -10 (if present)
    env.reset()
    if 'No Medication_active' in env.action_cols:
        a = np.zeros(len(env.action_cols), dtype=int)
        idx = env.action_cols.index('No Medication_active')
        a[idx] = 1
        a[(idx + 1) % len(env.action_cols)] = 1
        _, reward, done, _, info = env.step(a)
        assert done and reward == -10.0


def test_subject_age_progression(env: ALPACAEnv):
    obs0, _ = env.reset()
    age0 = pd.Series(obs0, index=env.observation_cols).get('subject_age', None)
    a = np.zeros(len(env.action_cols), dtype=int)
    obs1, _, _, _, _ = env.step(a + (np.eye(len(env.action_cols), dtype=int)[0]))
    age1 = pd.Series(obs1, index=env.observation_cols).get('subject_age', None)
    if age0 is not None and age1 is not None:
        assert pytest.approx(age0 + env.time_delta_val / 12.0, rel=1e-6, abs=1e-6) == age1


def test_reward_scaling_and_clipping(env: ALPACAEnv):
    # Use a valid base observation, adjust metric by +/- up to 3 within bounds
    metric = env.reward_metric
    assert metric in env.observation_cols
    prev, _ = env.reset()
    prev = prev.copy()
    i = env.observation_cols.index(metric)
    base = float(prev[i])

    # Determine safe positive/negative deltas within bounds
    lb = -np.inf
    ub = np.inf
    if getattr(env, 'variable_bounds', None) is not None and metric in env.variable_bounds.index:
        lb = float(env.variable_bounds.loc[metric, 'lower_bound'])
        ub = float(env.variable_bounds.loc[metric, 'upper_bound'])

    # Positive direction
    max_pos = min(3.0, ub - base)
    if max_pos > 1e-6:
        nxt = prev.copy()
        nxt[i] = base + max_pos
        r_pos = env.calculate_reward(prev, nxt)
        expected = float(np.clip((10.0 / 3.0) * max_pos, -10.0, 10.0))
        assert pytest.approx(expected, rel=1e-6, abs=1e-6) == r_pos

    # Negative direction
    max_neg = min(3.0, base - lb)
    if max_neg > 1e-6:
        nxt = prev.copy()
        nxt[i] = base - max_neg
        r_neg = env.calculate_reward(prev, nxt)
        expected = float(np.clip(-(10.0 / 3.0) * max_neg, -10.0, 10.0))
        assert pytest.approx(expected, rel=1e-6, abs=1e-6) == r_neg


def test_categorical_one_hot_groups(env: ALPACAEnv):
    # After one step, categorical groups should remain one-hot (sum to 1) where applicable
    obs, _ = env.reset()
    a = env.action_space.sample()
    obs2, _, done, _, _ = env.step(a)
    s = pd.Series(obs2, index=env.observation_cols)
    for _, cols in getattr(env, 'y_categorical_groups', {}).items():
        cols = [c for c in cols if c in s.index]
        if not cols:
            continue
        total = float(s[cols].sum())
        # Allow occasional absence if schema columns missing; otherwise expect one-hot
        assert pytest.approx(total, abs=1e-6) == 1.0


def test_model_input_order_matches_schema(env: ALPACAEnv, alpaca_dir: Path):
    # Ensure the env uses the same order as columns_schema.json model_input_cols
    import json
    schema_path = alpaca_dir / 'columns_schema.json'
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    schema_cols = schema.get('model_input_cols', [])
    assert env.model_input_cols == schema_cols


def test_sequence_autoregression_and_rollout(env: ALPACAEnv):
    # Short rollout to ensure sequence grows and step returns valid outputs
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    steps = 4
    prev_len = 0
    for t in range(steps):
        a = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(a)
        assert isinstance(obs, np.ndarray)
        assert np.isfinite(reward)
        assert isinstance(done, bool) and isinstance(truncated, bool)
        # sequence_length should increase by 1 each step
        assert info.get('sequence_length', 0) == prev_len + 1
        prev_len = info.get('sequence_length', prev_len)
        if done or truncated:
            break


def test_bounds_checker_logic(env: ALPACAEnv):
    # 1. A valid state should pass
    state, _ = env.reset()
    is_valid, violations = env.check_state_bounds(state)
    assert is_valid
    assert len(violations) == 0

    # 2. An invalid state should fail with details
    state_invalid = state.copy()
    # Pick a variable with known bounds to violate
    metric_to_violate = 'ADNI_MEM'  # This is usually well-defined
    if metric_to_violate not in env.observation_cols or env.variable_bounds is None or metric_to_violate not in env.variable_bounds.index:
        pytest.skip(f"Cannot test bounds violation; '{metric_to_violate}' has no defined bounds.")

    metric_idx = env.observation_cols.index(metric_to_violate)
    upper_bound = env.variable_bounds.loc[metric_to_violate, 'upper_bound']

    # Violate upper bound
    state_invalid[metric_idx] = upper_bound + 1.0
    is_valid, violations = env.check_state_bounds(state_invalid)
    assert not is_valid
    assert len(violations) == 1
    assert violations[0]['variable'] == metric_to_violate
    assert violations[0]['value'] == upper_bound + 1.0

    # Violate lower bound
    lower_bound = env.variable_bounds.loc[metric_to_violate, 'lower_bound']
    state_invalid[metric_idx] = lower_bound - 1.0
    is_valid, violations = env.check_state_bounds(state_invalid)
    assert not is_valid
    assert len(violations) == 1
    assert violations[0]['variable'] == metric_to_violate
    assert violations[0]['value'] == lower_bound - 1.0


def test_get_start_state_chooses_earliest_visit_per_subject(alpaca_dir: Path):
    """
    Tests that get_start_state correctly identifies and uses the earliest
    visit for a randomly selected subject.
    """
    # Use a fresh env for this test to manipulate its data
    env = ALPACAEnv(data_path=str(alpaca_dir))
    df = env.training_data.copy()

    # Find a subject with multiple visits to make the test meaningful
    subject_counts = df['subject_id'].value_counts()
    test_subject_id = subject_counts[subject_counts > 1].index[0]

    # Determine the true earliest (unscaled) time for this subject
    subj_df = df[df['subject_id'] == test_subject_id]
    
    time_col = env.time_col
    j = list(env.scaler_X.feature_names_in_).index(time_col)
    subj_df_unscaled_time = subj_df[time_col] * float(env.scaler_X.scale_[j]) + float(env.scaler_X.mean_[j])
    earliest_unscaled_time = subj_df_unscaled_time.min()
    
    # Get the row(s) corresponding to the earliest visit
    earliest_visit_mask = np.isclose(subj_df_unscaled_time, earliest_unscaled_time)
    earliest_visits_df = subj_df[earliest_visit_mask]

    # The logic in get_start_state will sample one of these if there are ties.
    # We will mock the sample to return the first one for a deterministic test.
    expected_visit_df = earliest_visits_df.head(1)
    
    # Calculate the expected unscaled state from this visit
    scaled_obs_series = expected_visit_df[env.observation_cols].iloc[0]
    scaled_obs_df = pd.DataFrame([scaled_obs_series])
    
    cont_in_scalerX = [c for c in env.observation_cols if c in getattr(env.scaler_X, 'feature_names_in_', [])]
    if cont_in_scalerX:
        unscaled_df = env.manage_state_scaling(scaled_obs_df[cont_in_scalerX], env.scaler_X, normalize=False)
        for c in cont_in_scalerX:
            scaled_obs_df[c] = unscaled_df[c]
    
    expected_unscaled_state = scaled_obs_df.iloc[0][env.observation_cols].values.astype(np.float32)

    # Mock subject sampling to pick our test subject and visit sampling to be deterministic
    with patch('pandas.core.series.Series.sample', return_value=pd.Series([test_subject_id])), \
         patch('pandas.core.frame.DataFrame.sample', return_value=expected_visit_df):
        # Reset the environment, which calls get_start_state
        state, _ = env.reset()
        
    # 1. Check that the internal time is set to the earliest visit for that subject
    assert np.isclose(env._months_since_bl, earliest_unscaled_time, atol=1e-4)
    
    # 2. Check that the returned state matches the unscaled state of the earliest visit
    np.testing.assert_allclose(state, expected_unscaled_state, rtol=1e-5, atol=1e-5)


def test_get_start_state_synthetic_sampling_without_subject_id(alpaca_dir: Path):
    """
    Tests that get_start_state generates a valid synthetic state when
    subject_id is not available in the training data.
    """
    env = ALPACAEnv(data_path=str(alpaca_dir))
    
    # Remove subject_id to trigger the fallback logic
    df_with_sid = env.training_data.copy()
    env.training_data = df_with_sid.drop(columns=['subject_id'])
    
    # Determine the set of earliest visits that the logic should be sampling from
    time_col = env.time_col
    initial_visits_df = env.training_data
    global_min_time = 0.0
    if time_col in env.training_data.columns:
        m = env.training_data[time_col].astype(float)
        # Need to unscale time to find the minimum correctly
        if time_col in getattr(env.scaler_X, 'feature_names_in_', []):
            j = list(env.scaler_X.feature_names_in_).index(time_col)
            m_unscaled = m * float(env.scaler_X.scale_[j]) + float(env.scaler_X.mean_[j])
        else:
            m_unscaled = m
        global_min_time = m_unscaled.min()
        earliest_mask = (m_unscaled - global_min_time).abs() <= 1e-6
        initial_visits_df = env.training_data[earliest_mask]

    # Reset the env to generate a synthetic state. Seed for reproducibility.
    state, _ = env.reset(seed=42)
    
    # 1. Check that the state is valid and within the defined observation space
    assert state is not None
    assert state.shape == env.observation_space.shape
    assert env.observation_space.contains(state), "Generated state is outside observation space bounds."
    
    # 2. Check that the internal time is set to the global minimum
    if time_col in env.training_data.columns:
        assert np.isclose(env._months_since_bl, global_min_time, atol=1e-4)

    # 3. Check that one-hot encoded groups sum to 1
    state_series = pd.Series(state, index=env.observation_cols)
    used_group_cols = set()
    for group_cols in getattr(env, 'y_categorical_groups', {}).values():
        cols = [c for c in group_cols if c in env.observation_cols]
        if not cols:
            continue
        assert np.isclose(state_series[cols].sum(), 1.0, atol=1e-6), f"Group {cols} does not sum to 1."
        used_group_cols.update(cols)

    # 4. For non-categorical, continuous columns that are sampled, verify the
    #    sampled value originates from the distribution of earliest visits.
    state_df = pd.DataFrame([state_series])
    
    # We need to re-scale the generated state to compare it to the original (scaled) training data
    scaled_state_df = env.manage_state_scaling(state_df, env.scaler_X, normalize=True)

    for col in env.observation_cols:
        if col in used_group_cols or col not in initial_visits_df.columns:
            continue
        
        # The generated value (after scaling it back) should be one of the
        # values present in the initial visits data for that column.
        generated_scaled_val = scaled_state_df[col].iloc[0]
        possible_values = initial_visits_df[col].unique()
        
        is_present = np.any(np.isclose(generated_scaled_val, possible_values))
        assert is_present, f"Value for '{col}' ({generated_scaled_val}) not in source distribution."


def test_step_terminates_on_out_of_bounds_next_state(env: ALPACAEnv):
    """
    Tests that the episode terminates if the model predicts a next state
    that is outside the defined variable bounds.
    """
    env.reset(seed=42)
    
    # Find a metric with a defined upper bound to violate
    metric_to_violate = 'ADNI_MEM'
    if metric_to_violate not in env.observation_cols or env.variable_bounds is None or metric_to_violate not in env.variable_bounds.index:
        pytest.skip(f"Cannot test bounds violation; '{metric_to_violate}' has no defined bounds.")

    upper_bound = env.variable_bounds.loc[metric_to_violate, 'upper_bound']

    # The model outputs scaled values. We need to produce a scaled value that,
    # when unscaled, will exceed the bound.
    # y_unscaled = y_scaled * scale + mean  => y_scaled = (y_unscaled - mean) / scale
    metric_idx_y = env.model_cont_output_cols.index(metric_to_violate)
    scaler_y_idx = list(env.scaler_y.feature_names_in_).index(metric_to_violate)
    mean = env.scaler_y.mean_[scaler_y_idx]
    scale = env.scaler_y.scale_[scaler_y_idx]
    
    # Target a value just above the upper bound
    violation_value_unscaled = upper_bound + 1.0
    violation_value_scaled = (violation_value_unscaled - mean) / scale

    # Mock the model's output
    mock_pred_cont = torch.zeros((1, 1, len(env.model_cont_output_cols)), device=env.device)
    mock_pred_cont[0, 0, metric_idx_y] = violation_value_scaled
    
    mock_pred_bin = torch.zeros((1, 1, len(env.model_binary_output_cols)), device=env.device)

    with patch.object(env.model, 'forward', return_value=(mock_pred_cont, mock_pred_bin)):
        action = env.action_space.sample()
        # The state before the step
        state_before = env.state.copy()
        next_state, reward, done, truncated, info = env.step(action)

        # The episode should be 'done'
        assert done, "Episode should terminate on out-of-bounds state."
        # Truncated should be False, as this is a terminal condition
        assert not truncated
        # Reward should be 0.0 for this type of termination
        assert reward == 0.0
        # Info should contain the reason
        assert info.get('termination_reason') == 'state_out_of_bounds'
        assert len(info.get('out_of_bounds_variables', [])) == 1
        violation_info = info['out_of_bounds_variables'][0]
        assert violation_info['variable'] == metric_to_violate
        assert np.isclose(violation_info['value'], violation_value_unscaled)
        
        # The state should not have been updated to the invalid state
        np.testing.assert_allclose(next_state, state_before, rtol=1e-6)