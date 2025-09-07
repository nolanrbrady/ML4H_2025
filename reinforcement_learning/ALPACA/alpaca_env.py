import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
import joblib
import torch
from moe_transformer import TransformerWithMoE



# ALzheimers Prophelactic Action Control Agent (ALPACA) Environment
class ALPACAEnv(gym.Env):
    def __init__(
        self,
        data_path='.',
        force_baseline_start=False,
        time_delta_months: float = 6.0,
        reward_metric: str = 'ADNI_MEM',
    ):
        """
        Initialize the ALPACA environment.

        Args:
            data_path (str): Path to the data files
            force_baseline_start (bool): Deprecated. The environment now always starts from the
                                       earliest available visit in the dataset (based on unscaled
                                       months_since_bl). This parameter is accepted but ignored.
            time_delta_months (float): Fixed step size in months passed to model as next_visit_months.
            reward_metric (str): Column name in observation used for reward delta (default: 'ADNI_MEM').
        """
        self.data_path = data_path
        # Deprecated flag retained for backward compatibility (ignored)
        self.force_baseline_start = force_baseline_start
        self.reward_metric = reward_metric

        # Estimated test-retest reliability coefficient for the ADNI_MEM variable
        self.reliability_rxx = 0.91

        super().__init__()

        # --- File Paths ---
        self.scaler_x_path = os.path.join(data_path, 'scaler_X.joblib')
        self.scaler_y_path = os.path.join(data_path, 'scaler_y.joblib')
        self.model_path = os.path.join(data_path, 'best_moe_transformer_model.pt')
        self.training_data_path = os.path.join(data_path, 'X_train.csv')
        self.bounds_path = os.path.join(data_path, 'ADNI_Variable_Bounds.csv')

        # --- Load Artifacts ---
        self.scaler_X = joblib.load(self.scaler_x_path)
        self.scaler_y = joblib.load(self.scaler_y_path)
        self.training_data = pd.read_csv(self.training_data_path)

        # --- Load Variable Bounds ---
        if os.path.exists(self.bounds_path):
            self.variable_bounds = pd.read_csv(self.bounds_path, index_col=0)
            # print(f"Loaded variable bounds from {self.bounds_path}")
        else:
            print(f"Warning: Variable bounds file not found at {self.bounds_path}")
            self.variable_bounds = None

        # --- Column Definitions (Prefer schema; fallback to training order) ---
        schema_path = os.path.join(data_path, 'columns_schema.json')
        schema = None
        if os.path.exists(schema_path):
            try:
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                print(f"Loaded columns_schema.json from {schema_path}")
            except Exception as e:
                print(f"Warning: Failed to read columns_schema.json: {e}")
                schema = None

        # Names for time bookkeeping
        self.time_col = 'months_since_bl'
        self.delta_col = 'next_visit_months'

        if schema:
            # Trust schema as source of truth for columns and order
            self.action_cols = list(schema.get('action_cols', []))
            self.observation_cols = list(schema.get('observation_cols', []))
            self.model_input_cols = list(schema.get('model_input_cols', self.observation_cols + self.action_cols + [self.delta_col]))
            # Model outputs (continuous scaled; binary probabilities)
            self.model_cont_output_cols = [c for c in schema.get('y_cont_cols', [])]
            self.model_binary_output_cols = [c for c in schema.get('y_bin_cols', [])]
            # For reference only
            self.binary_obs_cols = [c for c in self.observation_cols if c in set(self.model_binary_output_cols)]
            self.cont_obs_cols = [c for c in self.observation_cols if c not in set(self.binary_obs_cols)]
            self.y_categorical_groups = schema.get('y_categorical_groups', {})
        else:
            # Fallback: derive from training data, preserving existing behavior
            self.action_cols = sorted([col for col in self.training_data.columns if col.endswith('_active')])
            self.cont_obs_cols = sorted(['ADNI_MEM', 'ADNI_EF2', 'TAU_data', 'subject_age', 'ABETA', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV'])
            self.binary_obs_cols = sorted([col for col in self.training_data.columns if col.startswith(('PTGENDER_', 'PTRACCAT_'))])
            self.observation_cols = self.cont_obs_cols + self.binary_obs_cols
            self.model_input_cols = self.cont_obs_cols + [self.delta_col] + self.action_cols + self.binary_obs_cols
            self.model_cont_output_cols = self.cont_obs_cols
            self.model_binary_output_cols = self.binary_obs_cols
            self.y_categorical_groups = {}

        # --- Align scaler_X feature names to schema/model inputs ---
        self._align_scaler_X_feature_names()

        # --- Model Setup (MoE Transformer, autoregressive) ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")        # GPU
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")         # Apple Silicon (Metal)
        else:
            self.device = torch.device("cpu")
        
        self.model = TransformerWithMoE(
            input_dim=len(self.model_input_cols),
            out_cont_dim=len(self.model_cont_output_cols),
            out_bin_dim=len(self.model_binary_output_cols),
        ).to(self.device)
        try:
            state = torch.load(self.model_path, map_location=self.device)
            # Support both plain state_dict and checkpoint dicts
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            # Require exact key match so we fail fast on incompatibilities
            self.model.load_state_dict(state, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
        self.model.eval()

        # --- Environment State ---
        self.max_episode_length = 22
        self.time_delta_val = float(time_delta_months)
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.info = {}
        # Internal sequence buffer of scaled model inputs
        self._seq_inputs = []  # list of np.ndarray rows ordered by self.model_input_cols
        # Internal clock (unscaled)
        self._months_since_bl = 0.0
        # Bounds tolerance to avoid numerical edge false-positives
        self._bounds_eps = 1e-6

        # --- Gym Spaces ---
        self.action_space = gym.spaces.MultiBinary(len(self.action_cols))

        # Create observation space bounds based on unscaled (real-world) values
        obs_data_scaled = self.training_data
        lows = np.zeros(len(self.observation_cols), dtype=np.float32)
        highs = np.zeros(len(self.observation_cols), dtype=np.float32)
        for i, col in enumerate(self.observation_cols):
            if self.variable_bounds is not None and col in self.variable_bounds.index:
                lows[i] = float(self.variable_bounds.loc[col, 'lower_bound'])
                highs[i] = float(self.variable_bounds.loc[col, 'upper_bound'])
            elif col in getattr(self.scaler_X, 'feature_names_in_', []):
                # Unscale min/max from training_data using scaler_X stats
                col_min = obs_data_scaled[col].min()
                col_max = obs_data_scaled[col].max()
                j = list(self.scaler_X.feature_names_in_).index(col)
                lows[i] = float(col_min * self.scaler_X.scale_[j] + self.scaler_X.mean_[j])
                highs[i] = float(col_max * self.scaler_X.scale_[j] + self.scaler_X.mean_[j])
            else:
                # Binary 0/1 (or fallback to observed bounds)
                lows[i] = float(obs_data_scaled[col].min())
                highs[i] = float(obs_data_scaled[col].max())
        self.observation_space = gym.spaces.Box(
            low=lows, high=highs, shape=(len(self.observation_cols),), dtype=np.float32
        )

        self.state = self.reset()[0]

    def _align_scaler_X_feature_names(self):
        """Align scaler_X feature names to match schema/model input expectations.

        - Map any time-gap aliases (e.g., 'time_delta', 'time_since_prev') to
          the configured delta column 'next_visit_months'.
        - Drop absolute time like 'months_since_bl' which is not part of model inputs.
        - Remove any names not present in model_input_cols.
        - Avoid duplicates and preserve the first occurrence stats.
        """
        sx = getattr(self, 'scaler_X', None)
        if sx is None or not hasattr(sx, 'feature_names_in_'):
            return
        names = list(sx.feature_names_in_)
        means = np.array(getattr(sx, 'mean_', []), dtype=float)
        scales = np.array(getattr(sx, 'scale_', []), dtype=float) if hasattr(sx, 'scale_') else None
        vars_ = np.array(getattr(sx, 'var_', []), dtype=float) if hasattr(sx, 'var_') else None

        new_names: list[str] = []
        new_means: list[float] = []
        new_scales: list[float] = []
        new_vars: list[float] = []

        def maybe_append(target_name: str, i: int):
            if target_name in new_names:
                return
            if target_name not in self.model_input_cols:
                return
            new_names.append(target_name)
            if i < len(means):
                new_means.append(float(means[i]))
            if scales is not None and i < len(scales):
                new_scales.append(float(scales[i]))
            if vars_ is not None and i < len(vars_):
                new_vars.append(float(vars_[i]))

        for i, name in enumerate(names):
            if name in ('months_since_bl',):
                # Drop absolute time from input scaling
                continue
            if name in ('time_delta', 'time_since_prev'):
                maybe_append(self.delta_col, i)
            else:
                maybe_append(name, i)

        # If the scaler didn't have any time-gap but model expects one, leave it unscaled
        sx.feature_names_in_ = np.array(new_names, dtype=object)
        sx.n_features_in_ = len(new_names)
        if hasattr(sx, 'mean_'):
            sx.mean_ = np.array(new_means, dtype=float)
        if hasattr(sx, 'scale_') and len(new_scales) > 0:
            sx.scale_ = np.array(new_scales, dtype=float)
        if hasattr(sx, 'var_') and len(new_vars) > 0:
            sx.var_ = np.array(new_vars, dtype=float)

    def manage_state_scaling(self, state_data, scaler, normalize=True):
        """Scale/unscale only the intersection of columns present in both DataFrame and scaler.

        Avoids calling sklearn transformers with mismatched feature counts by
        applying the transformation per-column using the scaler's learned stats.
        """
        if not isinstance(state_data, pd.DataFrame):
            raise TypeError("Input 'state_data' must be a pandas DataFrame.")
        data = state_data.copy()
        # If scaler doesn't expose feature names, return unchanged
        if not hasattr(scaler, 'feature_names_in_'):
            return data

        name_to_idx = {c: i for i, c in enumerate(scaler.feature_names_in_)}
        with_mean = getattr(scaler, 'with_mean', True)
        with_std = getattr(scaler, 'with_std', True)

        for col in list(data.columns):
            if col not in name_to_idx:
                # Column wasn't seen by scaler; leave as-is
                continue
            j = name_to_idx[col]
            col_vals = data[col].astype(float)
            if normalize:
                # x' = (x - mean) / scale
                if with_mean:
                    col_vals = col_vals - float(scaler.mean_[j])
                if with_std:
                    scale_j = float(scaler.scale_[j]) if getattr(scaler, 'scale_', None) is not None else 1.0
                    if scale_j != 0.0:
                        col_vals = col_vals / scale_j
            else:
                # x = x' * scale + mean
                if with_std:
                    scale_j = float(scaler.scale_[j]) if getattr(scaler, 'scale_', None) is not None else 1.0
                    col_vals = col_vals * scale_j
                if with_mean:
                    col_vals = col_vals + float(scaler.mean_[j])
            data[col] = col_vals.astype(np.float32)
        return data

    def _inverse_scale_series(self, series: pd.Series, scaler, cols: list[str]) -> pd.Series:
        """Inverse-scale values for specified cols using scaler stats (per-column)."""
        out = series.copy()
        if not hasattr(scaler, 'feature_names_in_'):
            return out
        name_to_idx = {c: i for i, c in enumerate(scaler.feature_names_in_)}
        for c in cols:
            if c in name_to_idx and c in out.index:
                j = name_to_idx[c]
                out[c] = np.float32(out[c] * scaler.scale_[j] + scaler.mean_[j])
        return out

    def check_state_bounds(self, state_values):
        """
        Check if state values are within the acceptable ADNI variable bounds.

        Args:
            state_values (numpy.ndarray): Array of state values corresponding to observation_cols

        Returns:
            tuple: (is_within_bounds: bool, out_of_bounds_variables: list)
        """
        if self.variable_bounds is None:
            # If no bounds file loaded, assume all values are valid
            return True, []

        state_series = pd.Series(state_values, index=self.observation_cols)
        out_of_bounds_vars = []

        for var_name, value in state_series.items():
            # Skip variables not in bounds file (e.g., binary demographic variables)
            if var_name not in self.variable_bounds.index:
                continue

            lower_bound = float(self.variable_bounds.loc[var_name, 'lower_bound'])
            upper_bound = float(self.variable_bounds.loc[var_name, 'upper_bound'])

            # Allow tiny numerical tolerance at the edges
            if value < (lower_bound - self._bounds_eps) or value > (upper_bound + self._bounds_eps):
                out_of_bounds_vars.append({
                    'variable': var_name,
                    'value': value,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })

        is_within_bounds = len(out_of_bounds_vars) == 0
        return is_within_bounds, out_of_bounds_vars

    def get_start_state(self):
        """Select a random patient, then their earliest visit (unscaled).

        - Randomly samples a `subject_id` from the dataset.
        - Computes unscaled `months_since_bl` for that subject's rows.
        - Chooses the earliest-visit row (ties broken randomly).
        - Returns the unscaled observation vector for that row.
        """
        df = self.training_data
        sid_col = 'subject_id'

        # Fallback if subject_id is missing: revert to global earliest
        if sid_col not in df.columns:
            # Last resort: sample a synthetic start state from feature-wise distributions
            # Prefer the distribution at the earliest global time if available
            if self.time_col in df.columns:
                m = df[self.time_col].astype(float)
                if self.time_col in getattr(self.scaler_X, 'feature_names_in_', []):
                    j = list(self.scaler_X.feature_names_in_).index(self.time_col)
                    m_unscaled = m * float(self.scaler_X.scale_[j]) + float(self.scaler_X.mean_[j])
                else:
                    m_unscaled = m
                min_time = float(m_unscaled.min())
                tol = 1e-6
                earliest_mask = (m_unscaled - min_time).abs() <= tol
                initial_visits = df[earliest_mask]
                if initial_visits.empty:
                    initial_visits = df
            else:
                initial_visits = df

            # Build a scaled synthetic observation row by sampling each feature
            scaled_vals = {}
            # First handle grouped one-hot binaries by sampling a category per group
            used_group_cols = set()
            for group_cols in getattr(self, 'y_categorical_groups', {}).values():
                cols = [c for c in group_cols if c in self.observation_cols]
                if not cols:
                    continue
                # Empirical probabilities from earliest distribution
                probs = initial_visits[cols].mean(numeric_only=True)
                # Normalize to avoid degenerate cases
                if probs.sum() <= 0 or probs.isna().any():
                    # fallback to uniform
                    probs = pd.Series([1.0/len(cols)]*len(cols), index=cols)
                else:
                    probs = probs / probs.sum()
                winner = np.random.choice(probs.index, p=probs.values)
                for c in cols:
                    scaled_vals[c] = 1.0 if c == winner else 0.0
                used_group_cols.update(cols)

            # Handle remaining observation columns
            for c in self.observation_cols:
                if c in used_group_cols:
                    continue
                if c in initial_visits.columns:
                    # Sample an existing value
                    try:
                        val = initial_visits[c].sample(n=1).iloc[0]
                    except Exception:
                        val = initial_visits[c].iloc[0]
                    scaled_vals[c] = float(val) if pd.api.types.is_numeric_dtype(initial_visits[c]) else val
                else:
                    # Default zeros if missing
                    scaled_vals[c] = 0.0

            scaled_state_df = pd.DataFrame([scaled_vals])
            # Set internal time to earliest if available
            if self.time_col in df.columns:
                self._months_since_bl = float(min_time)
            else:
                self._months_since_bl = 0.0
        else:
            # Sample a patient id uniformly across unique subjects (avoid visit-count bias)
            unique_sids = df[sid_col].drop_duplicates()
            sid = unique_sids.sample(n=1).iloc[0]
            subj_df = df[df[sid_col] == sid]

            # If time column is missing, pick any row for this subject
            if self.time_col not in subj_df.columns or subj_df.empty:
                sampled_visit = subj_df.sample(n=1) if not subj_df.empty else df.sample(n=1)
            else:
                m = subj_df[self.time_col].astype(float)
                if self.time_col in getattr(self.scaler_X, 'feature_names_in_', []):
                    j = list(self.scaler_X.feature_names_in_).index(self.time_col)
                    m_unscaled = m * float(self.scaler_X.scale_[j]) + float(self.scaler_X.mean_[j])
                else:
                    m_unscaled = m
                min_time = float(m_unscaled.min())
                tol = 1e-6
                earliest_mask = (m_unscaled - min_time).abs() <= tol
                initial_visits = subj_df[earliest_mask]
                if initial_visits.empty:
                    # Fallback to absolute min row
                    initial_visits = subj_df.iloc[[int(m_unscaled.idxmin())]]
                sampled_visit = initial_visits.sample(n=1)

        # Build scaled observation (obs only), then inverse-scale per-column to unscaled space
        if 'scaled_state_df' not in locals():
            scaled_start_state = sampled_visit[self.observation_cols].iloc[0].copy()
            scaled_state_df = pd.DataFrame([scaled_start_state])

        cont_in_scalerX = [c for c in self.observation_cols if c in getattr(self.scaler_X, 'feature_names_in_', [])]
        if cont_in_scalerX:
            unscaled = self.manage_state_scaling(scaled_state_df[cont_in_scalerX], self.scaler_X, normalize=False)
            for c in cont_in_scalerX:
                scaled_state_df[c] = unscaled[c]

        # Track months_since_bl in unscaled units if using a concrete sampled_visit
        if 'sampled_visit' in locals() and isinstance(sampled_visit, pd.DataFrame) and self.time_col in sampled_visit.columns:
            m_scaled = float(sampled_visit[self.time_col].iloc[0])
            if self.time_col in getattr(self.scaler_X, 'feature_names_in_', []):
                j = list(self.scaler_X.feature_names_in_).index(self.time_col)
                self._months_since_bl = m_scaled * float(self.scaler_X.scale_[j]) + float(self.scaler_X.mean_[j])
            else:
                self._months_since_bl = m_scaled

        # Return unscaled observation
        unscaled_start_state = scaled_state_df.iloc[0][self.observation_cols].copy()
        return unscaled_start_state.values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._seq_inputs = []
        self.state = self.get_start_state()
        self.done = False
        self.reward = 0
        self.info = {}
        return self.state, self.info

    def calculate_reward(self, prev_obs, next_obs):
        """
        Reliable-change reward (bounded to [-10, 10]):

            r_t = clip( 10 * Δ / S_diff, -10, 10 )

        where Δ = S_{t+1} - S_t for the selected reward_metric, and
            S_diff = sqrt(2 * (1 - r_xx)) * SD.

        Assumptions:
        - ADNI_MEM is z-scaled ⇒ SD = 1.
        - Test–retest reliability r_xx ≈ 0.9 (can override via self.reliability_rxx).

        If the next state violates bounds, reward is neutralized to 0.
        """
        prev_series = pd.Series(prev_obs, index=self.observation_cols)
        next_series = pd.Series(next_obs, index=self.observation_cols)
        metric = self.reward_metric
        if metric not in prev_series.index:
            return 0.0

        # Change in (z-)units
        delta = float(next_series[metric] - prev_series[metric])

        # Reliable change denominator: S_diff = sqrt(2*(1 - r_xx)) * SD
        r_xx = float(getattr(self, "reliability_rxx", 0.9))
        r_xx = float(np.clip(r_xx, 0.0, 0.999999))  # numerical safety
        sd = 1.0  # ADNI_MEM is z-scaled
        s_diff = float(np.sqrt(2.0 * (1.0 - r_xx)) * sd)
        if not np.isfinite(s_diff) or s_diff <= 1e-12:
            s_diff = 0.4472135955  # ≈ sqrt(0.2) fallback (r_xx ~ 0.9, SD=1)

        # Map 1 reliable-change unit to ±10 reward and clip
        scaled = 10.0 * (delta / s_diff)
        base_reward = float(np.clip(scaled, -10.0, 10.0))

        # Neutralize reward if bounds violated (episode will terminate)
        within_bounds, _ = self.check_state_bounds(next_obs)
        if not within_bounds:
            base_reward = 0.0

        return base_reward

    def step(self, action):
        if self.done:
            return self.state, self.reward, self.done, True, self.info
        
        # Initialize info dictionary for this step
        self.info = {}

        # Ensure that if no_medication action is taken, then no other action can be taken
        # This enforces medical validity - no medication should be mutually exclusive
        no_med_idx = None
        try:
            no_med_idx = self.action_cols.index('No Medication_active')
        except ValueError:
            # Handle case where 'No Medication_active' column doesn't exist
            print("Warning: 'No Medication_active' column not found in action columns")

        if no_med_idx is not None and action[no_med_idx] == 1:
            # Check if any other action is also taken
            other_actions_active = np.sum(action) > 1  # Sum > 1 means other actions are also active
            if other_actions_active:
                # Violation of no-medication constraint
                # print(f"Action constraint violation: No Medication selected with other treatments.")
                # print(f"Action vector: {action}")

                # End episode with negative reward
                self.done = True
                self.reward = -10.0  # Strong negative reward for constraint violation
                # Reflect attempted step in sequence length even if invalid
                self.info = {
                    'constraint_violation': 'no_medication_with_other_actions',
                    'sequence_length': len(self._seq_inputs) + 1,
                }
                return self.state, self.reward, self.done, True, self.info

        # Check if no action is taken at all - penalize inaction
        if np.sum(action) == 0:
            # Violation of action requirement constraint
            # print(f"Action constraint violation: No action taken at all.")
            # print(f"Action vector: {action}")

            # End episode with negative reward
            self.done = True
            self.reward = -10.0  # Strong negative reward for taking no action
            self.info = {
                'constraint_violation': 'no_action_taken',
                'sequence_length': len(self._seq_inputs) + 1,
            }
            return self.state, self.reward, self.done, True, self.info

        self.current_step += 1
        current_obs_series = pd.Series(self.state, index=self.observation_cols)
        action_series = pd.Series(action, index=self.action_cols)
        # 1. Prepare model input with correct column order
        model_input_series = pd.concat([current_obs_series, action_series])
        model_input_series[self.delta_col] = self.time_delta_val
        model_input_df = pd.DataFrame([model_input_series])[self.model_input_cols]

        # 2. Scale only the features that scaler_X was trained on (intersection)
        cols_to_scale = [c for c in getattr(self.scaler_X, 'feature_names_in_', []) if c in model_input_df.columns]
        if cols_to_scale:
            scaled_subset = self.manage_state_scaling(model_input_df[cols_to_scale], self.scaler_X, normalize=True)
            # Assign per-column as float32 to avoid pandas dtype warnings
            for c in cols_to_scale:
                model_input_df[c] = np.float32(scaled_subset[c].values)

        # 3. Append to sequence buffer and form (B, S, F)
        self._seq_inputs.append(model_input_df.values.astype(np.float32)[0])
        x = torch.tensor(np.stack(self._seq_inputs, axis=0)[None, ...], dtype=torch.float32, device=self.device)

        # 4. Causal attention mask
        S = x.shape[1]
        attn_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=self.device), diagonal=1)

        # 5. Model prediction for last step
        pred_cont, pred_bin = self.model(x, attn_mask=attn_mask)
        last_cont = pred_cont[:, -1, :] if pred_cont is not None else None
        last_bin = pred_bin[:, -1, :] if pred_bin is not None else None

        # 6. Process predictions -> DataFrames (scaled space)
        if last_cont is not None:
            cont_vals = last_cont.detach().cpu().numpy()[0]
            cont_scaled_series = pd.Series(cont_vals, index=self.model_cont_output_cols)
        else:
            cont_scaled_series = pd.Series(dtype=float)
        if last_bin is not None:
            bin_probs = torch.sigmoid(last_bin).detach().cpu().numpy()[0]
            bin_series = pd.Series(bin_probs, index=self.model_binary_output_cols)
        else:
            bin_series = pd.Series(dtype=float)

        # 7. Inverse-scale continuous predictions to real-world
        cont_unscaled = self._inverse_scale_series(cont_scaled_series, self.scaler_y, self.model_cont_output_cols)

        # 8. Construct next observation and enforce categorical one-hot groups
        next_obs_series = current_obs_series.copy()
        for col, value in cont_unscaled.items():
            if col in next_obs_series.index:
                next_obs_series[col] = np.float32(value)
        # Binary groups: choose argmax within each group
        used_bin_cols = set()
        for group_cols in getattr(self, 'y_categorical_groups', {}).values():
            cols = [c for c in group_cols if c in self.model_binary_output_cols]
            if not cols:
                continue
            probs = bin_series[cols].astype(float)
            if len(probs) == 0:
                continue
            winner = probs.idxmax()
            for c in cols:
                if c in next_obs_series.index:
                    next_obs_series[c] = 1.0 if c == winner else 0.0
            used_bin_cols.update(cols)
        # Any remaining independent binary outputs: round to nearest {0,1}
        for c, p in bin_series.items():
            if c not in used_bin_cols and c in next_obs_series.index:
                next_obs_series[c] = float(np.round(p))

        # 9. Update time and subject age deterministically
        prev_months = self._months_since_bl
        self._months_since_bl = prev_months + self.time_delta_val
        if 'subject_age' in next_obs_series.index:
            next_obs_series['subject_age'] = float(current_obs_series.get('subject_age', 0.0)) + (self.time_delta_val / 12.0)

        # 10. Check if next state is within acceptable bounds
        next_state_np = next_obs_series[self.observation_cols].values
        is_within_bounds, out_of_bounds_vars = self.check_state_bounds(next_state_np)

        if not is_within_bounds:
            # State has drifted outside acceptable bounds - terminate episode without negative reward
            self.done = True
            self.reward = 0.0  # Neutral reward - not the agent's fault, just model drift
            self.info = {
                'termination_reason': 'state_out_of_bounds',
                'out_of_bounds_variables': out_of_bounds_vars
            }
            # Don't update state - keep current state
            truncated = False  # This is a done condition, not truncation
            return self.state, self.reward, self.done, truncated, self.info

        # 11. Finalize and update (only if within bounds)
        self.reward = self.calculate_reward(self.state, next_state_np)
        self.state = next_state_np
        self.info = {
            'sequence_length': len(self._seq_inputs),
        }

        if self.current_step >= self.max_episode_length:
            self.done = True

        truncated = self.current_step >= self.max_episode_length
        return self.state, self.reward, self.done, truncated, self.info

    def render(self, render_mode='None'):
        pass

    def close(self):
        pass


# Test the environment
def main():
    try:
        # Get the directory where the script is located
        # script_dir = os.path.dirname(os.path.realpath(__file__))
        env = ALPACAEnv()
        obs, info = env.reset()
        print("--- Initial Observation (unscaled) ---")
        print(pd.Series(obs, index=env.observation_cols))

        for i in range(5):
            action = env.action_space.sample()
            print(f"\n--- Step {i+1} ---")
            print("Action taken:", action.tolist())

            next_state, reward, done, truncated, info = env.step(action)

            print("\n--- Next Observation (unscaled) ---")
            print(pd.Series(next_state, index=env.observation_cols))
            print(f"Reward: {reward:.4f}")
            print(f"Done: {done}, Truncated: {truncated}")

            if done or truncated:
                print("\nEpisode finished.")
                obs, info = env.reset()
                print("\n--- Resetting Environment ---")
                print("--- Initial Observation (unscaled) ---")
                print(pd.Series(obs, index=env.observation_cols))


    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
