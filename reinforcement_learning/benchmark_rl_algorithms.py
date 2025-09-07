#!/usr/bin/env python3
"""
RL Algorithm Comparison Script (Shared Framework)
================================================

Compares trained RL algorithms (PPO, SAC, A2C, DQN) on the ALPACA environment
using the shared evaluation framework. All algorithms are evaluated on the same
initial patient states for fair comparison.

Evaluates:
- Reward differences between algorithms
- Cognitive metric progression (TRABSCOR or ADNI_MEM depending on env)
- Episode performance metrics
- Statistical significance of differences
- Action selection patterns

Usage:
    python benchmark_rl_algorithms.py [--num_episodes N] [--output_dir DIR]
"""

import os
import sys
import argparse
import warnings
from typing import Dict, Any

warnings.filterwarnings('ignore')

from stable_baselines3 import PPO, SAC, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    # If running as a package module
    from .evaluation_framework import PolicyEvaluator, SB3PolicyAdapter, NoMedicationPolicyAdapter
except Exception:
    # If executed as a script
    from evaluation_framework import PolicyEvaluator, SB3PolicyAdapter, NoMedicationPolicyAdapter


def find_latest_models() -> Dict[str, Dict[str, str]]:
    """Find latest trained models for each algorithm under this directory."""
    algorithms_config: Dict[str, Dict[str, str]] = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))

    algorithm_patterns = {
        'PPO': ['ppo_alpaca_'],
        'SAC': ['sac_alpaca_'],
        'A2C': ['a2c_alpaca_'],
        'DQN': ['dqn_alpaca_'],
    }

    for alg_name, patterns in algorithm_patterns.items():
        latest_dir = None
        latest_stamp = ''
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if not os.path.isdir(item_path):
                continue
            for pattern in patterns:
                if item.startswith(pattern):
                    stamp = item.replace(pattern, '').split('_')[0]
                    model_path = os.path.join(item_path, 'best_model', 'best_model.zip')
                    if stamp > latest_stamp and os.path.exists(model_path):
                        latest_dir = item
                        latest_stamp = stamp
        if latest_dir:
            model_path = os.path.join(current_dir, latest_dir, 'best_model', 'best_model.zip')
            vec_path = os.path.join(current_dir, latest_dir, 'vec_normalize.pkl')
            algorithms_config[alg_name] = {
                'model_path': model_path,
                'vec_normalize_path': vec_path if os.path.exists(vec_path) else None,
            }
            print(f"Found {alg_name} model: {latest_dir}")

    return algorithms_config


def main():
    parser = argparse.ArgumentParser(description='Compare RL algorithms on ALPACA environment')
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='rl_algorithm_comparison_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic_only', action='store_true',
                        help='Run only deterministic evaluation (skip stochastic).')
    parser.add_argument('--algorithms', type=str, nargs='*', default=None,
                        help='Subset of algorithms to compare: PPO SAC A2C DQN')
    args = parser.parse_args()

    # Discover models
    print('Searching for trained RL models...')
    algorithms_config = find_latest_models()
    if not algorithms_config:
        print('Error: No trained RL models found!')
        return

    if args.algorithms:
        req = set(a.upper() for a in args.algorithms)
        algorithms_config = {k: v for k, v in algorithms_config.items() if k.upper() in req}
        if not algorithms_config:
            print('Error: None of the requested algorithms are available!')
            return

    print(f"Will compare algorithms: {list(algorithms_config.keys())}")

    # Prepare policies
    current_dir = os.path.dirname(os.path.abspath(__file__))
    alpaca_path = os.path.join(current_dir, 'ALPACA')
    if alpaca_path not in sys.path:
        sys.path.append(alpaca_path)
    from alpaca_env import ALPACAEnv  # type: ignore

    policies: Dict[str, Any] = {}
    for alg_name, cfg in algorithms_config.items():
        model_path = cfg.get('model_path')
        algo_cls = PPO if alg_name.upper() == 'PPO' else SAC if alg_name.upper() == 'SAC' else A2C if alg_name.upper() == 'A2C' else DQN if alg_name.upper() == 'DQN' else None
        if algo_cls is None or not model_path or not os.path.exists(model_path):
            print(f"Warning: skipping {alg_name}; model not found or class unresolved")
            continue
        model = algo_cls.load(model_path)

        normalizer = None
        vec_path = cfg.get('vec_normalize_path')
        if vec_path and os.path.exists(vec_path):
            # Use the ALPACA assets folder as data_path so required artifacts are found
            dummy_env = DummyVecEnv([lambda: ALPACAEnv(data_path=alpaca_path, force_baseline_start=False)])
            normalizer = VecNormalize.load(vec_path, dummy_env)
            normalizer.training = False
            print(f"Loaded VecNormalize for {alg_name} from {vec_path}")

        policies[alg_name] = SB3PolicyAdapter(name=alg_name, model=model, normalizer=normalizer)

    # Add NoMed baseline
    # Initialize a temporary env to read action columns, using explicit data_path
    tmp_env = ALPACAEnv(data_path=alpaca_path, force_baseline_start=False)
    policies.setdefault('NoMed', NoMedicationPolicyAdapter(tmp_env.action_cols))

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

    print(f"\nComparison completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
