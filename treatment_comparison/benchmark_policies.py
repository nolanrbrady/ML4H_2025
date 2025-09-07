#!/usr/bin/env python3
"""
Policy Benchmarking (Shared Framework)
=====================================

Benchmarks a PPO agent against the clinician policy on the ALPACA environment
using the shared evaluation framework. Both policies are evaluated on the same
initial patient states for fair comparison.

Usage:
    python benchmark_policies.py [--num_episodes N] [--output_dir DIR]
"""

import os
import sys
import argparse
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

warnings.filterwarnings('ignore')

# Add ALPACA path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
alpaca_path = os.path.join(parent_dir, 'reinforcement_learning', 'ALPACA')
if alpaca_path not in sys.path:
    sys.path.append(alpaca_path)
from alpaca_env import ALPACAEnv  # type: ignore

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from bayesian_model import BayesianModel  # noqa: E402


class ClinicianPolicy:
    def __init__(self, model_path: str, scaler_path: str):
        self.device = torch.device('cpu')
        self.scaler = joblib.load(scaler_path)
        print(f"Loaded clinician scaler from {scaler_path}")
        # Infer input size from scaler feature names
        self.feature_names = list(getattr(self.scaler, 'feature_names_in_', []))
        input_size = len(self.feature_names) if self.feature_names else 0
        if input_size == 0:
            raise ValueError("Clinician scaler does not expose feature_names_in_; cannot determine input size.")
        num_binary_outputs = 17
        num_continuous_outputs = 0
        self.model = BayesianModel(
            input_size=input_size,
            num_continuous_outputs=num_continuous_outputs,
            num_binary_outputs=num_binary_outputs,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded clinician model from {model_path}")
        print(f"Clinician model expects {input_size} features (aligned to scaler feature_names_in_)")

    def _assemble_input(self, state: np.ndarray, env: ALPACAEnv) -> np.ndarray:
        """Map env observation + known time fields into the clinician feature order, then scale.

        - Fills any missing features with sensible defaults (0.0)
        - Handles time features: months_since_bl and next_visit_months/time_delta
        - Applies scaler normalization using per-column stats
        """
        # Build a dict for the single-row input
        obs_idx = {name: i for i, name in enumerate(getattr(env, 'observation_cols', []))}
        row = {}
        for name in self.feature_names:
            if name in obs_idx:
                row[name] = float(state[obs_idx[name]])
            elif name in ('months_since_bl', 'MonthsSinceBL'):
                row[name] = float(getattr(env, '_months_since_bl', 0.0))
            elif name in ('next_visit_months', 'time_delta', 'time_since_prev'):
                row[name] = float(getattr(env, 'time_delta_val', 0.0))
            else:
                row[name] = 0.0
        # Order columns as in feature_names
        import pandas as pd  # local import to avoid global dependency at module import
        df = pd.DataFrame([[row[n] for n in self.feature_names]], columns=self.feature_names)
        # Normalize using clinician scaler (per-column stats)
        if hasattr(env, 'manage_state_scaling'):
            df_scaled = env.manage_state_scaling(df, self.scaler, normalize=True)
        else:
            df_scaled = df
        return df_scaled.values.astype(np.float32)[0]

    def predict(self, state: np.ndarray, env: ALPACAEnv, use_topk_sampling: bool = True) -> np.ndarray:
        features = self._assemble_input(state, env)
        input_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, binary_probs = self.model.predict(input_tensor)
            binary_probs_np = binary_probs.cpu().numpy()[0]

        if not use_topk_sampling:
            threshold = 0.5
            binary_actions = (binary_probs_np > threshold).astype(int)
            try:
                no_med_idx = env.action_cols.index('No Medication_active')
            except ValueError:
                return binary_actions
            if binary_actions[no_med_idx] == 1 and np.sum(binary_actions) > 1:
                no_med_prob = binary_probs_np[no_med_idx]
                other_mask = binary_actions.copy()
                other_mask[no_med_idx] = 0
                if np.any(other_mask):
                    other_max = np.max(binary_probs_np[other_mask == 1])
                    if no_med_prob > other_max:
                        binary_actions = np.zeros_like(binary_actions)
                        binary_actions[no_med_idx] = 1
                    else:
                        binary_actions[no_med_idx] = 0
            if np.sum(binary_actions) == 0:
                binary_actions[np.argmax(binary_probs_np)] = 1
            return binary_actions

        # Top-k sampling path with No-Med constraint
        binary_actions = np.zeros_like(binary_probs_np, dtype=int)
        try:
            no_med_idx = env.action_cols.index('No Medication_active')
        except ValueError:
            no_med_idx = None
        if no_med_idx is not None:
            highest_prob_idx = np.argmax(binary_probs_np)
            if highest_prob_idx == no_med_idx:
                binary_actions[no_med_idx] = 1
                return binary_actions
            else:
                available_actions = np.arange(len(binary_probs_np))
                available_actions = available_actions[available_actions != no_med_idx]
                available_probs = binary_probs_np[available_actions]
        else:
            available_actions = np.arange(len(binary_probs_np))
            available_probs = binary_probs_np

        high_prob_mask = available_probs > 0.6
        if np.any(high_prob_mask):
            binary_actions[available_actions[high_prob_mask]] = 1
        min_threshold = 0.15
        candidate_mask = available_probs > min_threshold
        if np.any(candidate_mask):
            candidate_probs = available_probs[candidate_mask]
            candidate_indices = available_actions[candidate_mask]
            sorted_indices = candidate_indices[np.argsort(candidate_probs)[::-1]]
            num_actions_to_select = min(4, len(sorted_indices))
            if len(sorted_indices) > 2:
                weights = candidate_probs[np.argsort(candidate_probs)[::-1]][:num_actions_to_select]
                weights = weights / weights.sum()
                num_to_select = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
                selected_indices = np.random.choice(
                    sorted_indices[:num_actions_to_select],
                    size=min(num_to_select, len(sorted_indices)),
                    replace=False,
                    p=weights[: len(sorted_indices)],
                )
                binary_actions[selected_indices] = 1
            else:
                binary_actions[sorted_indices] = 1
        if np.sum(binary_actions) == 0:
            if len(available_actions) > 0:
                highest_prob_idx = available_actions[np.argmax(available_probs)]
                binary_actions[highest_prob_idx] = 1
            else:
                binary_actions[np.argmax(binary_probs_np)] = 1
        return binary_actions


# -------- Shared evaluation framework wiring ------------------------------------

rl_dir = os.path.join(parent_dir, 'reinforcement_learning')
if rl_dir not in sys.path:
    sys.path.append(rl_dir)
try:
    from evaluation_framework import (
        PolicyEvaluator,
        SB3PolicyAdapter,
        ClinicianPolicyAdapter,
        NoMedicationPolicyAdapter,
    )
except Exception:
    from autoreg.reinforcement_learning.evaluation_framework import (  # type: ignore
        PolicyEvaluator,
        SB3PolicyAdapter,
        ClinicianPolicyAdapter,
        NoMedicationPolicyAdapter,
    )


def _find_latest_models(search_dir: str):
    """Find the latest PPO, SAC, and A2C models within `search_dir`.

    Returns a dict mapping algorithm name -> {model_path, vec_normalize_path}.
    """
    patterns = {
        'PPO': ['ppo_alpaca_'],
        'SAC': ['sac_alpaca_'],
        'A2C': ['a2c_alpaca_'],
    }

    latest_by_alg = {alg: {'stamp': '', 'dir': None} for alg in patterns.keys()}

    for entry in os.listdir(search_dir):
        entry_path = os.path.join(search_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        for alg, prefixes in patterns.items():
            for pref in prefixes:
                if entry.startswith(pref):
                    stamp = entry.replace(pref, '').split('_')[0]
                    model_zip = os.path.join(entry_path, 'best_model', 'best_model.zip')
                    if os.path.exists(model_zip) and stamp >= latest_by_alg[alg]['stamp']:
                        latest_by_alg[alg] = {'stamp': stamp, 'dir': entry}

    found = {}
    for alg, info in latest_by_alg.items():
        if info['dir'] is None:
            continue
        model_dir = os.path.join(search_dir, info['dir'])
        model_path = os.path.join(model_dir, 'best_model', 'best_model.zip')
        vec_path = os.path.join(model_dir, 'vec_normalize.pkl')
        found[alg] = {
            'model_path': model_path,
            'vec_normalize_path': vec_path if os.path.exists(vec_path) else None,
        }
    return found


def main():
    parser = argparse.ArgumentParser(description='Benchmark RL Agents vs Clinician (shared framework)')
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='benchmark_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic_only', action='store_true',
                        help='Run only deterministic evaluation (skip stochastic).')
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(__file__)
    models = _find_latest_models(base_dir)
    if not models:
        print("Error: No PPO/SAC/A2C models found in the current directory.")
        return
    clinician_model_path = os.path.join(base_dir, 'best_clinician_policy.pth')
    clinician_scaler_path = os.path.join(base_dir, 'scaler_clinician_X.joblib')

    if not os.path.exists(clinician_model_path):
        print(f"Error: Clinician model not found at {clinician_model_path}")
        return
    if not os.path.exists(clinician_scaler_path):
        print(f"Error: Clinician scaler not found at {clinician_scaler_path}")
        return

    # Build policy set for PPO/SAC/A2C
    policies = {}
    for alg_name, cfg in models.items():
        mp = cfg.get('model_path')
        vp = cfg.get('vec_normalize_path')
        if alg_name == 'PPO':
            model = PPO.load(mp)
        elif alg_name == 'SAC':
            model = SAC.load(mp)
        elif alg_name == 'A2C':
            model = A2C.load(mp)
        else:
            continue
        print(f"Loaded {alg_name} model from {mp}")
        normalizer = None
        if vp and os.path.exists(vp):
            dummy_env = DummyVecEnv([lambda: ALPACAEnv(data_path=alpaca_path, force_baseline_start=False)])
            normalizer = VecNormalize.load(vp, dummy_env)
            normalizer.training = False
            print(f"Loaded VecNormalize for {alg_name} from {vp}")
        policies[alg_name] = SB3PolicyAdapter(name=alg_name, model=model, normalizer=normalizer)

    # Clinician model and adapter
    clinician = ClinicianPolicy(model_path=clinician_model_path, scaler_path=clinician_scaler_path)
    policies['Clinician'] = ClinicianPolicyAdapter(
        name='Clinician', clinician_policy=clinician, deterministic_topk=False, stochastic_topk=True
    )

    # No Medication baseline
    tmp_env = ALPACAEnv(data_path=alpaca_path, force_baseline_start=False)
    policies['No Medication'] = NoMedicationPolicyAdapter(tmp_env.action_cols)

    evaluator = PolicyEvaluator(alpaca_path=alpaca_path, policies=policies)
    evaluator.run_comparison(
        num_episodes=args.num_episodes,
        seed=args.seed,
        do_deterministic=True,
        do_stochastic=not args.deterministic_only,
    )
    analysis = evaluator.analyze_results(output_dir=args.output_dir)

    # Console summaries and tests (deterministic and stochastic if present)
    for ct in ['deterministic', 'stochastic']:
        if ct not in analysis:
            continue
        comp = analysis[ct]
        print(f"\n{ct.title()} Summary:")
        for name, s in comp.get('summary', {}).items():
            print(
                f"  {name:>12}: reward={s['mean_reward']:.3f} | "
                f"{evaluator.cognitive_metric}={s['mean_final_cognitive_metric']:.3f} | "
                f"len={s['mean_episode_length']:.2f}"
            )
        if comp.get('statistical_tests'):
            print(f"\nPairwise tests ({ct.title()}):")
            for pair, tests in comp['statistical_tests'].items():
                for test_name, vals in tests.items():
                    stat = vals.get('statistic')
                    p = vals.get('pvalue')
                    sig = vals.get('significant')
                    md = vals.get('mean_diff')
                    extra = f", diff={md:.3f}" if md is not None else ""
                    print(f"  {pair} {test_name}: stat={stat:.3f}, p={p:.6f}{extra} ({'Sig' if sig else 'NS'})")

    print(f"\nBenchmark completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
