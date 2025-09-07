#!/usr/bin/env python3
"""
DQN Benchmark for ALPACA Environment
====================================

Deep Q-Network (DQN) benchmark script for the ALPACA 
(Alzheimer's Prophylactic Action Control Agent) environment.

This script trains a DQN agent and provides comprehensive logging and 
monitoring to understand training performance and convergence.

Note: DQN requires discrete action spaces. We convert the MultiBinary 
action space to a single Discrete space with 2^n possible combinations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import warnings
import itertools
from scipy import stats
warnings.filterwarnings('ignore')

# Add ALPACA path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ALPACA'))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium.spaces import Discrete

from alpaca_env import ALPACAEnv


class MultiBinaryToDiscreteWrapper(gym.Wrapper):
    """
    Wrapper to convert MultiBinary action space to Discrete for DQN compatibility.
    DQN requires discrete action spaces, so we map all 2^n combinations to discrete actions.
    
    Warning: This can create very large action spaces (2^17 = 131,072 for ALPACA).
    We implement a more practical subset approach.
    """
    
    def __init__(self, env, max_actions=1024):
        super().__init__(env)
        self.n_binary_actions = env.action_space.n
        self.max_actions = max_actions
        
        # Generate a subset of meaningful action combinations
        # Instead of all 2^n combinations, we use a more practical approach
        self.action_mapping = self._generate_action_mapping()
        
        # Create discrete action space
        self.action_space = Discrete(len(self.action_mapping))
        
        print(f"Converted MultiBinary({self.n_binary_actions}) to Discrete({len(self.action_mapping)})")
    
    def _generate_action_mapping(self):
        """Generate a practical subset of action combinations"""
        action_mapping = []
        
        # Add all-zeros action (no treatment)
        action_mapping.append(np.zeros(self.n_binary_actions, dtype=int))
        
        # Add single-action activations (each treatment individually)
        for i in range(self.n_binary_actions):
            action = np.zeros(self.n_binary_actions, dtype=int)
            action[i] = 1
            action_mapping.append(action)
        
        # Add some random combinations up to max_actions
        np.random.seed(42)  # For reproducibility
        while len(action_mapping) < min(self.max_actions, 2**self.n_binary_actions):
            # Generate random action with 1-5 active treatments
            n_active = np.random.randint(1, min(6, self.n_binary_actions + 1))
            action = np.zeros(self.n_binary_actions, dtype=int)
            active_indices = np.random.choice(self.n_binary_actions, n_active, replace=False)
            action[active_indices] = 1
            
            # Check if this combination is already in the mapping
            if not any(np.array_equal(action, existing) for existing in action_mapping):
                action_mapping.append(action)
        
        return action_mapping
    
    def step(self, action):
        # Convert discrete action to binary action
        # Ensure action is a scalar integer
        if hasattr(action, '__len__') and len(action) == 1:
            # Handle single-element arrays (like numpy arrays)
            action_idx = int(action[0])
        elif hasattr(action, 'item'):
            # Handle numpy scalars
            action_idx = int(action.item())
        else:
            # Handle regular Python integers
            action_idx = int(action)
        
        binary_action = self.action_mapping[action_idx]
        return self.env.step(binary_action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def get_action_description(self, action_idx):
        """Get human-readable description of action"""
        binary_action = self.action_mapping[action_idx]
        active_treatments = [self.env.action_cols[i] for i in range(len(binary_action)) if binary_action[i] == 1]
        if not active_treatments:
            return "No treatment"
        return f"Treatments: {', '.join(active_treatments)}"


class TrainingMonitorCallback(BaseCallback):
    """Custom callback for detailed training monitoring"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_mean_reward = -np.inf
        self.action_counts = {}
        
    def _on_step(self) -> bool:
        # Track actions taken
        if hasattr(self.locals, 'actions') and self.locals['actions'] is not None:
            for action in self.locals['actions']:
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
        
        # Check if episode is done
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    # Get episode statistics
                    if 'episode' in self.locals['infos'][i]:
                        episode_info = self.locals['infos'][i]['episode']
                        episode_reward = episode_info['r']
                        episode_length = episode_info['l']
                        
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        self.episode_count += 1
                        
                        # Print every 10 episodes
                        if self.episode_count % 10 == 0:
                            mean_reward = np.mean(self.episode_rewards[-100:])
                            mean_length = np.mean(self.episode_lengths[-100:])
                            
                            print(f"\nEpisode {self.episode_count}")
                            print(f"  Mean Reward (last 100): {mean_reward:.4f}")
                            print(f"  Mean Length (last 100): {mean_length:.2f}")
                            print(f"  Current Episode Reward: {episode_reward:.4f}")
                            print(f"  Total Timesteps: {self.num_timesteps}")
                            
                            if mean_reward > self.best_mean_reward:
                                self.best_mean_reward = mean_reward
                                print(f"  🎉 New best mean reward: {self.best_mean_reward:.4f}")
                            
                            # Show action distribution
                            if self.action_counts:
                                top_actions = sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                                print(f"  Top actions: {[(a, c) for a, c in top_actions]}")
        
        return True


def create_alpaca_env():
    """Create and return ALPACA environment with Discrete action space wrapper"""
    try:
        # Change to ALPACA directory to access data files
        original_dir = os.getcwd()
        alpaca_dir = os.path.join(os.path.dirname(__file__), 'ALPACA')
        os.chdir(alpaca_dir)
        
        # Create environment
        env = ALPACAEnv(data_path='.', force_baseline_start=False)
        
        # Wrap with MultiBinary to Discrete converter for DQN
        env = MultiBinaryToDiscreteWrapper(env, max_actions=512)  # Limit action space size
        
        # Return to original directory
        os.chdir(original_dir)
        
        return env
    except Exception as e:
        print(f"Error creating ALPACA environment: {e}")
        raise


def analyze_environment():
    """Analyze the ALPACA environment structure"""
    print("=" * 80)
    print("ALPACA ENVIRONMENT ANALYSIS (DQN with MultiBinary→Discrete)")
    print("=" * 80)
    
    env = create_alpaca_env()
    
    print(f"Observation Space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Low: {env.observation_space.low}")
    print(f"  High: {env.observation_space.high}")
    
    print(f"\nAction Space (Wrapped for DQN): {env.action_space}")
    print(f"  Number of discrete actions: {env.action_space.n}")
    print(f"  Original MultiBinary Size: {env.n_binary_actions}")
    print(f"  Theoretical max combinations: {2**env.n_binary_actions:,}")
    print(f"  Practical subset used: {len(env.action_mapping)}")
    
    print(f"\nSample Action Descriptions:")
    for i in range(min(10, len(env.action_mapping))):
        print(f"  Action {i:2d}: {env.get_action_description(i)}")
    
    print(f"\nObservation Columns ({len(env.env.observation_cols)}):")
    for i, col in enumerate(env.env.observation_cols):
        print(f"  {i:2d}: {col}")
    
    print(f"\nOriginal Action Columns ({len(env.env.action_cols)}):")
    for i, col in enumerate(env.env.action_cols):
        print(f"  {i:2d}: {col}")
    
    print(f"\nEpisode Settings:")
    print(f"  Max Episode Length: {env.env.max_episode_length}")
    print(f"  Time Delta: {env.env.time_delta_val} months")
    
    # Check environment compatibility
    try:
        check_env(env)
        print(f"\n✅ Environment passes Gym compatibility checks")
    except Exception as e:
        print(f"\n⚠️ Environment compatibility warning: {e}")
    
    # Test a few random episodes
    print(f"\nTesting Random Episodes:")
    total_rewards = []
    episode_lengths = []
    action_usage = {}
    
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\n  Episode {episode + 1}:")
        print(f"    Initial TRABSCOR: {obs[env.env.observation_cols.index('TRABSCOR')]:.2f}")
        print(f"    Initial Age: {obs[env.env.observation_cols.index('subject_age')]:.1f}")
        
        while True:
            action = env.action_space.sample()
            action_usage[action] = action_usage.get(action, 0) + 1
            
            if steps == 0:  # Show first action
                print(f"    First Action: {action} - {env.get_action_description(action)}")
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if steps <= 3:  # Show first few steps
                print(f"    Step {steps}: Reward={reward:.4f}, TRABSCOR={obs[env.env.observation_cols.index('TRABSCOR')]:.2f}")
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"    Total Reward: {episode_reward:.4f}, Steps: {steps}")
    
    print(f"\nRandom Episode Statistics:")
    print(f"  Mean Reward: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"  Actions used: {len(action_usage)} out of {env.action_space.n}")
    
    env.close()
    return env.observation_space, env.action_space


def train_dqn_agent(total_timesteps=50000, learning_rate=1e-4, buffer_size=50000, batch_size=32):
    """Train DQN agent on ALPACA environment"""
    
    print("=" * 80)
    print("DQN TRAINING ON ALPACA ENVIRONMENT")
    print("=" * 80)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"dqn_alpaca_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create tensorboard directory
    tensorboard_dir = f"tensorboard/dqn_{timestamp}"
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    print(f"Training Configuration:")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Buffer Size: {buffer_size:,}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Model Directory: {model_dir}")
    
    # Create vectorized environment
    def make_env():
        return create_alpaca_env()
    
    vec_env = make_vec_env(make_env, n_envs=1, seed=42)  # DQN typically uses single env
    
    # Normalize observations but not rewards (rewards are already bounded)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Create evaluation environment
    eval_env = make_vec_env(make_env, n_envs=1, seed=123)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=1.0,  # Hard update
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,  # Explore for 30% of training
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        learning_starts=1000,
        tensorboard_log=tensorboard_dir,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=42
    )
    
    # Set up logging
    new_logger = configure(f"{model_dir}/logs/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Create callbacks
    monitor_callback = TrainingMonitorCallback(verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model/",
        log_path=f"{model_dir}/eval_logs/",
        eval_freq=max(5000, buffer_size // 10),
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=10
    )
    
    callback_list = CallbackList([monitor_callback, eval_callback])
    
    print(f"\nStarting DQN training...")
    print(f"Monitor training progress with: tensorboard --logdir {tensorboard_dir}")
    print(f"Note: DQN will explore randomly at the beginning (epsilon-greedy)")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        
        # Save final model
        model.save(f"{model_dir}/final_model")
        vec_env.save(f"{model_dir}/vec_normalize.pkl")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Final model saved to: {model_dir}/final_model")
        
        # Training summary
        if len(monitor_callback.episode_rewards) > 0:
            print(f"\nTraining Summary:")
            print(f"  Total Episodes: {monitor_callback.episode_count}")
            print(f"  Best Mean Reward: {monitor_callback.best_mean_reward:.4f}")
            print(f"  Final Mean Reward (last 100): {np.mean(monitor_callback.episode_rewards[-100:]):.4f}")
            print(f"  Mean Episode Length: {np.mean(monitor_callback.episode_lengths):.2f}")
            
            # Action distribution analysis
            if monitor_callback.action_counts:
                print(f"\nAction Usage Analysis:")
                total_actions = sum(monitor_callback.action_counts.values())
                top_actions = sorted(monitor_callback.action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for action, count in top_actions:
                    percentage = (count / total_actions) * 100
                    print(f"  Action {action:3d}: {count:5d} times ({percentage:5.1f}%)")
        
        return model, vec_env, model_dir, monitor_callback, timestamp
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


def evaluate_random_policy(n_episodes=50):
    """Evaluate random policy baseline for comparison"""
    print("=" * 80)
    print("EVALUATING RANDOM POLICY BASELINE")
    print("=" * 80)
    
    env = create_alpaca_env()
    episode_rewards = []
    episode_lengths = []
    medical_outcomes = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Get initial TRABSCOR
        trabscor_idx = env.env.observation_cols.index('TRABSCOR')
        initial_trabscor = obs[trabscor_idx]
        final_trabscor = initial_trabscor
        
        while True:
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Update final TRABSCOR
            final_trabscor = obs[trabscor_idx]
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        trabscor_change = final_trabscor - initial_trabscor
        medical_outcomes.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_length,
            'trabscor_change': trabscor_change,
            'initial_trabscor': initial_trabscor,
            'final_trabscor': final_trabscor
        })
        
        if episode < 5:  # Show details for first 5 episodes
            print(f"Episode {episode + 1:2d}: Reward={episode_reward:8.4f}, Length={episode_length:2d}, TRABSCOR: {initial_trabscor:.2f}→{final_trabscor:.2f} (Δ={trabscor_change:+.3f})")
    
    env.close()
    
    # Summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_trabscor_change = np.mean([outcome['trabscor_change'] for outcome in medical_outcomes])
    
    print(f"\nRandom Policy Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Mean Episode Length: {mean_length:.2f}")
    print(f"  Mean TRABSCOR Change: {mean_trabscor_change:.4f} (Note: Higher TRABSCOR = Worse cognition)")
    print(f"  Reward Range: [{min(episode_rewards):.4f}, {max(episode_rewards):.4f}]")
    
    # Interpret TRABSCOR changes for random policy
    if mean_trabscor_change > 0:
        print(f"  🔴 Random policy causes cognitive decline: +{mean_trabscor_change:.2f} TRABSCOR points")
    elif mean_trabscor_change < 0:
        print(f"  🟢 Random policy shows cognitive improvement: {mean_trabscor_change:.2f} TRABSCOR points")
    else:
        print(f"  ⚪ Random policy shows no cognitive change")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'mean_trabscor_change': mean_trabscor_change,
        'episode_rewards': episode_rewards,
        'medical_outcomes': medical_outcomes
    }


def evaluate_trained_model(model, vec_env, random_baseline, n_episodes=20):
    """Evaluate the trained DQN model"""
    
    print("=" * 80)
    print("EVALUATING TRAINED DQN MODEL")
    print("=" * 80)
    
    episode_rewards = []
    episode_lengths = []
    medical_outcomes = []
    action_usage = {}
    
    # Get the wrapped environment for action descriptions
    # Need to dig through the wrapper stack to find our MultiBinaryToDiscreteWrapper
    wrapped_env = vec_env.envs[0]
    while hasattr(wrapped_env, 'env') and not hasattr(wrapped_env, 'action_mapping'):
        wrapped_env = wrapped_env.env
    
    # Create a separate non-normalized environment for TRABSCOR tracking
    raw_env = create_alpaca_env()
    
    for episode in range(n_episodes):
        # Reset both environments
        obs = vec_env.reset()
        raw_obs, _ = raw_env.reset()
        
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        
        # Get initial TRABSCOR from raw (unnormalized) environment
        trabscor_idx = raw_env.env.observation_cols.index('TRABSCOR')
        initial_trabscor = raw_obs[trabscor_idx]
        final_trabscor = initial_trabscor
        
        while True:
            # Get action from trained model using normalized observations
            action, _states = model.predict(obs, deterministic=True)
            action_idx = int(action[0])  # Ensure we have a scalar integer
            episode_actions.append(action_idx)
            action_usage[action_idx] = action_usage.get(action_idx, 0) + 1
            
            # Step both environments with the same action
            obs, reward, done, info = vec_env.step(action)
            raw_obs, raw_reward, raw_done, raw_truncated, raw_info = raw_env.step(wrapped_env.action_mapping[action_idx])  # Convert back to binary
            
            # Track TRABSCOR from raw environment
            final_trabscor = raw_obs[trabscor_idx]
            
            episode_reward += reward[0]
            episode_length += 1
            
            if done[0] or raw_done or raw_truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Analyze most frequent action in episode
        most_common_action = max(set(episode_actions), key=episode_actions.count)
        
        # Get the actual binary action and treatment description
        binary_action = wrapped_env.action_mapping[most_common_action]
        treatment_description = wrapped_env.get_action_description(most_common_action)
        # Find the base ALPACA environment for action column names
        base_env = wrapped_env
        while hasattr(base_env, 'env') and not hasattr(base_env, 'action_cols'):
            base_env = base_env.env
        active_treatments = [base_env.action_cols[i] for i in range(len(binary_action)) if binary_action[i] == 1]
        
        trabscor_change = final_trabscor - initial_trabscor
        medical_outcomes.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_length,
            'trabscor_change': trabscor_change,
            'initial_trabscor': initial_trabscor,
            'final_trabscor': final_trabscor,
            'most_common_action_idx': most_common_action,
            'most_common_action_binary': binary_action.tolist(),
            'most_common_action_treatments': active_treatments,
            'treatment_description': treatment_description,
            'unique_actions': len(set(episode_actions)),
            'action_sequence': episode_actions
        })
        
        if episode < 5:  # Show details for first 5 episodes
            print(f"Episode {episode + 1:2d}: Reward={episode_reward:8.4f}, Length={episode_length:2d}, TRABSCOR: {initial_trabscor:.2f}→{final_trabscor:.2f} (Δ={trabscor_change:+.3f})")
            print(f"              Most common action: {most_common_action} = {treatment_description}")
            print(f"              Binary representation: {binary_action}")
            print(f"              Unique actions used: {len(set(episode_actions))}")
    
    raw_env.close()
    
    # Summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_trabscor_change = np.mean([outcome['trabscor_change'] for outcome in medical_outcomes])
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Mean Episode Length: {mean_length:.2f}")
    print(f"  Mean TRABSCOR Change: {mean_trabscor_change:.4f} (Note: Higher TRABSCOR = Worse cognition)")
    print(f"  Reward Range: [{min(episode_rewards):.4f}, {max(episode_rewards):.4f}]")
    
    # Interpret TRABSCOR changes
    if mean_trabscor_change > 0:
        print(f"  🔴 Average cognitive decline: TRABSCOR increased by {mean_trabscor_change:.2f} points")
    elif mean_trabscor_change < 0:
        print(f"  🟢 Average cognitive improvement: TRABSCOR decreased by {abs(mean_trabscor_change):.2f} points")
    else:
        print(f"  ⚪ No average cognitive change")
    
    print(f"\nAction Usage Analysis:")
    total_actions = sum(action_usage.values())
    used_actions = len(action_usage)
    top_actions = sorted(action_usage.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"  Actions used: {used_actions} out of {wrapped_env.action_space.n}")
    print(f"  Top 5 actions:")
    for action, count in top_actions:
        percentage = (count / total_actions) * 100
        print(f"    Action {action:3d}: {count:4d} times ({percentage:5.1f}%)")
    
    # Compare against random baseline
    baseline_mean = random_baseline['mean_reward']
    baseline_std = random_baseline['std_reward']
    
    # Calculate improvement metrics
    reward_improvement = mean_reward - baseline_mean
    percent_improvement = (reward_improvement / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
    
    # Statistical significance (simple t-test approximation)
    baseline_rewards = random_baseline['episode_rewards']
    try:
        t_stat, p_value = stats.ttest_ind(episode_rewards, baseline_rewards)
        significant = p_value < 0.05
    except:
        t_stat, p_value, significant = 0, 1.0, False
    
    print(f"\nComparison vs Random Baseline:")
    print(f"  Random Policy Mean: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"  Trained Model Mean: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Absolute Improvement: {reward_improvement:+.4f}")
    print(f"  Percent Improvement: {percent_improvement:+.2f}%")
    print(f"  Statistical Significance: {'Yes' if significant else 'No'} (p={p_value:.4f})")
    
    # Performance assessment
    if reward_improvement > 2 * baseline_std:
        performance_level = "EXCELLENT"
        assessment = f"Strong improvement over random ({percent_improvement:+.1f}%)"
    elif reward_improvement > baseline_std:
        performance_level = "GOOD"
        assessment = f"Clear improvement over random ({percent_improvement:+.1f}%)"
    elif reward_improvement > 0:
        performance_level = "MODERATE"
        assessment = f"Some improvement over random ({percent_improvement:+.1f}%)"
    else:
        performance_level = "POOR"
        assessment = f"No improvement over random ({percent_improvement:+.1f}%)"
    
    print(f"  Performance Assessment: {performance_level}")
    print(f"  Summary: {assessment}")
    
    return episode_rewards, medical_outcomes, {
        'improvement': reward_improvement,
        'percent_improvement': percent_improvement,
        'significant': significant,
        'p_value': p_value,
        'performance_level': performance_level,
        'assessment': assessment
    }


def decode_action(action_idx, wrapper_env):
    """Utility function to decode what a discrete action index means"""
    if hasattr(wrapper_env, 'action_mapping'):
        binary_action = wrapper_env.action_mapping[action_idx]
        description = wrapper_env.get_action_description(action_idx)
        active_treatments = [wrapper_env.env.action_cols[i] for i in range(len(binary_action)) if binary_action[i] == 1]
        
        print(f"Action {action_idx}:")
        print(f"  Binary: {binary_action}")
        print(f"  Description: {description}")
        print(f"  Active treatments: {active_treatments}")
        return binary_action, description, active_treatments
    else:
        print(f"Action {action_idx}: Raw action (no wrapper)")
        return action_idx, f"Raw action {action_idx}", []


def main():
    """Main benchmark function"""
    print("🧠 DQN Benchmark for ALPACA Environment")
    print("=" * 80)
    
    try:
        # Step 1: Analyze environment
        obs_space, action_space = analyze_environment()
        
        # Step 2: Establish random policy baseline
        random_baseline = evaluate_random_policy(n_episodes=50)
        
        # Step 3: Train DQN agent
        model, vec_env, model_dir, monitor_callback, timestamp = train_dqn_agent(
            total_timesteps=100000,  # DQN often needs many timesteps
            learning_rate=1e-4,
            buffer_size=50000,
            batch_size=32
        )
        
        # Step 4: Evaluate trained model
        episode_rewards, medical_outcomes, comparison_stats = evaluate_trained_model(model, vec_env, random_baseline, n_episodes=50)
        
        # Step 5: Save results
        results_df = pd.DataFrame(medical_outcomes)
        results_df.to_csv(f"{model_dir}/evaluation_results.csv", index=False)
        
        # Save comparison statistics
        baseline_df = pd.DataFrame(random_baseline['medical_outcomes'])
        baseline_df.to_csv(f"{model_dir}/random_baseline_results.csv", index=False)
        
        # Save comparison summary
        comparison_df = pd.DataFrame([comparison_stats])
        comparison_df.to_csv(f"{model_dir}/performance_comparison.csv", index=False)
        
        print(f"\n✅ DQN Benchmark completed successfully!")
        print(f"📊 Results saved to: {model_dir}/")
        print(f"📈 View training progress: tensorboard --logdir ./tensorboard")
        
        # Final assessment based on comparison
        performance_level = comparison_stats['performance_level']
        assessment = comparison_stats['assessment']
        improvement = comparison_stats['percent_improvement']
        significant = comparison_stats['significant']
        
        if performance_level == "EXCELLENT":
            print(f"🎉 EXCELLENT: {assessment}")
            if significant:
                print(f"   ✅ Statistically significant improvement!")
        elif performance_level == "GOOD":
            print(f"✅ GOOD: {assessment}")
            if significant:
                print(f"   ✅ Statistically significant improvement!")
        elif performance_level == "MODERATE":
            print(f"⚠️  MODERATE: {assessment}")
            if not significant:
                print(f"   ⚠️  Improvement not statistically significant")
        else:
            print(f"❌ POOR: {assessment}")
            print("   Consider: longer training, different hyperparameters, or environment modifications")
            print("   Note: Large discrete action space may make learning challenging")
        
        # Action space analysis
        unique_actions_used = len(set(results_df['most_common_action_idx']))
        total_actions_available = action_space.n
        print(f"\nAction Space Utilization:")
        print(f"  Unique actions used: {unique_actions_used} out of {total_actions_available}")
        print(f"  Utilization rate: {(unique_actions_used/total_actions_available)*100:.1f}%")
        
        # Show most popular treatment combinations
        print(f"\nMost Popular Treatment Combinations:")
        treatment_counts = results_df['treatment_description'].value_counts().head(5)
        for i, (treatment, count) in enumerate(treatment_counts.items(), 1):
            print(f"  {i}. {treatment} (used {count} times)")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
