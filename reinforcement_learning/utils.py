"""
Shared utilities for RL algorithms on the ALPACA environment.

This module centralizes common logic used by A2C, PPO, and SAC scripts:
- Environment creation (with optional wrappers)
- Simple environment introspection printing
- Common training monitor callback
- Baseline/random policy evaluation
- Trained model evaluation with comparison to baseline
- VecEnv creation with normalization and logger setup helpers
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize


# ---------- Generic helpers ----------

def _alpaca_dir() -> str:
    return os.path.join(os.path.dirname(__file__), 'ALPACA')


def create_alpaca_env(
    reward_metric: Optional[str] = 'ADNI_MEM',
    force_baseline_start: bool = False,
    wrapper: Optional[Callable] = None,
    wrapper_kwargs: Optional[Dict] = None,
):
    """
    Factory to create a single ALPACA environment, optionally wrapped.

    - Handles working directory switching so ALPACA can read local data.
    - Optionally applies a `wrapper(env, **wrapper_kwargs)` around the env.
    - `reward_metric` is forwarded to ALPACAEnv if provided.
    """
    # Lazy import to avoid polluting sys.path at import-time of this module
    original_dir = os.getcwd()
    alpaca_dir = _alpaca_dir()
    sys.path.append(alpaca_dir)
    try:
        os.chdir(alpaca_dir)
        from alpaca_env import ALPACAEnv  # type: ignore

        env_kwargs = {
            'data_path': '.',
            'force_baseline_start': force_baseline_start,
        }
        if reward_metric is not None:
            env_kwargs['reward_metric'] = reward_metric

        env = ALPACAEnv(**env_kwargs)
        if wrapper is not None:
            wrapper_kwargs = wrapper_kwargs or {}
            env = wrapper(env, **wrapper_kwargs)
        return env
    finally:
        os.chdir(original_dir)


def timestamped_dir(prefix: str) -> Tuple[str, str]:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{ts}", ts


def configure_model_logger(model, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)


def make_vec_envs(env_fn: Callable, n_envs: int = 1, *,
                  norm_obs: bool = True, norm_reward: bool = False,
                  clip_obs: float = 10.0, gamma: float = 0.99):
    vec = make_vec_env(env_fn, n_envs=n_envs, seed=42)
    vec = VecNormalize(vec, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=clip_obs, gamma=gamma)
    return vec


# ---------- Feature / env inspection ----------

def get_underlying_env(env):
    """Unwraps nested gym wrappers to reach the base ALPACA env."""
    base = env
    # Gymnasium wraps expose .env; stop when no deeper env exists
    while hasattr(base, 'env') and base.env is not base:
        base = base.env
    return base


def feature_index(env, feature_name: str) -> int:
    base = get_underlying_env(env)
    return base.observation_cols.index(feature_name)


def print_env_overview(env, *, title: str = "ALPACA ENVIRONMENT ANALYSIS", target_feature: Optional[str] = None):
    print("=" * 80)
    print(title)
    print("=" * 80)

    base = get_underlying_env(env)
    print(f"Observation Space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    try:
        print(f"  Low: {env.observation_space.low}")
        print(f"  High: {env.observation_space.high}")
    except Exception:
        pass

    print(f"\nAction Space: {env.action_space}")
    if hasattr(env.action_space, 'n'):
        print(f"  Number of actions: {env.action_space.n}")
    elif hasattr(env.action_space, 'shape'):
        print(f"  Shape: {env.action_space.shape}")

    print(f"\nObservation Columns ({len(base.observation_cols)}):")
    for i, col in enumerate(base.observation_cols):
        print(f"  {i:2d}: {col}")

    print(f"\nAction Columns ({len(base.action_cols)}):")
    for i, col in enumerate(base.action_cols):
        print(f"  {i:2d}: {col}")

    print(f"\nEpisode Settings:")
    print(f"  Max Episode Length: {base.max_episode_length}")
    print(f"  Time Delta: {base.time_delta_val} months")

    # Quick random rollout
    print(f"\nTesting Random Episodes:")
    rewards = []
    lengths = []
    for ep in range(2):
        obs, info = env.reset()
        ep_r = 0
        steps = 0
        if target_feature is not None:
            idx = feature_index(env, target_feature)
            print(f"\n  Episode {ep + 1}:")
            print(f"    Initial {target_feature}: {obs[idx]:.3f}")
        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            ep_r += reward
            steps += 1
            if done or truncated:
                break
        rewards.append(ep_r)
        lengths.append(steps)
    print(f"\nRandom Episode Statistics:")
    print(f"  Mean Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"  Mean Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")


# ---------- Callbacks ----------

class TrainingMonitorCallback(BaseCallback):
    """Common episode tracking/printing for on-policy algorithms."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done and 'episode' in self.locals['infos'][i]:
                    ep = self.locals['infos'][i]['episode']
                    r = ep['r']
                    l = ep['l']
                    self.episode_rewards.append(r)
                    self.episode_lengths.append(l)
                    self.episode_count += 1
                    if self.episode_count % 10 == 0:
                        mean_r = float(np.mean(self.episode_rewards[-100:]))
                        mean_l = float(np.mean(self.episode_lengths[-100:]))
                        print(f"\nEpisode {self.episode_count}")
                        print(f"  Mean Reward (last 100): {mean_r:.4f}")
                        print(f"  Mean Length (last 100): {mean_l:.2f}")
                        print(f"  Current Episode Reward: {float(r):.4f}")
                        print(f"  Total Timesteps: {self.num_timesteps}")
                        if mean_r > self.best_mean_reward:
                            self.best_mean_reward = mean_r
                            print(f"  New best mean reward: {self.best_mean_reward:.4f}")
        return True


# ---------- Evaluation helpers ----------

def evaluate_random_policy_common(env_fn: Callable[[], object], n_episodes: int, *, target_feature: str) -> Dict:
    env = env_fn()
    episode_rewards = []
    episode_lengths = []
    medical_outcomes = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        ep_r = 0
        ep_len = 0

        idx = feature_index(env, target_feature)
        start_feat = obs[idx]
        last_feat = start_feat

        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            ep_r += reward
            ep_len += 1
            last_feat = obs[idx]
            if done or truncated:
                break

        episode_rewards.append(ep_r)
        episode_lengths.append(ep_len)
        medical_outcomes.append({
            'episode': episode + 1,
            'reward': ep_r,
            'length': ep_len,
            f'{target_feature.lower()}_change': last_feat - start_feat,
            f'initial_{target_feature.lower()}': start_feat,
            f'final_{target_feature.lower()}': last_feat,
        })

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    mean_feat_change = float(np.mean([o[f'{target_feature.lower()}_change'] for o in medical_outcomes]))

    print(f"\nRandom Policy Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Mean Episode Length: {mean_length:.2f}")
    print(f"  Mean {target_feature} Change: {mean_feat_change:.4f}")

    env.close()
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        f'mean_{target_feature.lower()}_change': mean_feat_change,
        'episode_rewards': episode_rewards,
        'medical_outcomes': medical_outcomes,
    }


def comparison_to_baseline(episode_rewards, random_baseline: Dict) -> Dict:
    from scipy import stats
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    baseline_mean = random_baseline['mean_reward']
    baseline_std = random_baseline['std_reward']
    improvement = mean_reward - baseline_mean
    percent_improvement = (improvement / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0.0
    try:
        t_stat, p_value = stats.ttest_ind(episode_rewards, random_baseline['episode_rewards'])
        significant = bool(p_value < 0.05)
    except Exception:
        p_value, significant = 1.0, False

    if improvement > 2 * baseline_std:
        level = "EXCELLENT"
        assessment = f"Strong improvement over random ({percent_improvement:+.1f}%)"
    elif improvement > baseline_std:
        level = "GOOD"
        assessment = f"Clear improvement over random ({percent_improvement:+.1f}%)"
    elif improvement > 0:
        level = "MODERATE"
        assessment = f"Some improvement over random ({percent_improvement:+.1f}%)"
    else:
        level = "POOR"
        assessment = f"No improvement over random ({percent_improvement:+.1f}%)"

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'improvement': improvement,
        'percent_improvement': percent_improvement,
        'significant': significant,
        'p_value': float(p_value),
        'performance_level': level,
        'assessment': assessment,
    }


def evaluate_trained_model_common(
    model,
    vec_env,
    env_fn: Callable[[], object],
    random_baseline: Dict,
    n_episodes: int,
    *,
    target_feature: str,
    collect_action_stats: bool = False,
):
    episode_rewards = []
    episode_lengths = []
    medical_outcomes = []
    action_statistics = [] if collect_action_stats else None

    raw_env = env_fn()

    for episode in range(n_episodes):
        obs = vec_env.reset()
        raw_obs, _ = raw_env.reset()
        ep_r = 0
        ep_len = 0
        if collect_action_stats:
            ep_actions = []

        idx = feature_index(raw_env, target_feature)
        start_feat = raw_obs[idx]
        last_feat = start_feat

        while True:
            action, _ = model.predict(obs, deterministic=True)
            if collect_action_stats:
                ep_actions.append(action[0].copy())
            obs, reward, done, info = vec_env.step(action)
            raw_obs, raw_reward, raw_done, raw_trunc, raw_info = raw_env.step(action[0])
            last_feat = raw_obs[idx]
            ep_r += reward[0]
            ep_len += 1
            if done[0] or raw_done or raw_trunc:
                break

        episode_rewards.append(ep_r)
        episode_lengths.append(ep_len)
        record = {
            'episode': episode + 1,
            'reward': ep_r,
            'length': ep_len,
            f'{target_feature.lower()}_change': last_feat - start_feat,
            f'initial_{target_feature.lower()}': start_feat,
            f'final_{target_feature.lower()}': last_feat,
        }
        if collect_action_stats:
            mean_actions = np.mean(np.array(ep_actions), axis=0)
            action_statistics.append(mean_actions)
            record['mean_action_usage'] = mean_actions.tolist()
        medical_outcomes.append(record)

    stats = comparison_to_baseline(episode_rewards, random_baseline)
    return episode_rewards, medical_outcomes, stats, action_statistics
