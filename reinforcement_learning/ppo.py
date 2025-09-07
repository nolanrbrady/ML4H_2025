#!/usr/bin/env python3
"""
PPO Benchmark for ALPACA Environment
=====================================

Proximal Policy Optimization (PPO) benchmark script for the ALPACA 
(Alzheimer's Prophylactic Action Control Agent) environment.

This script trains a PPO agent and provides comprehensive logging and 
monitoring to understand training performance and convergence.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

try:
    from .utils import (
        TrainingMonitorCallback,
        create_alpaca_env,
        print_env_overview,
        make_vec_envs,
        configure_model_logger,
        evaluate_random_policy_common,
        evaluate_trained_model_common,
    )
except ImportError:
    # Fallback for running as a script
    from utils import (
        TrainingMonitorCallback,
        create_alpaca_env,
        print_env_overview,
        make_vec_envs,
        configure_model_logger,
        evaluate_random_policy_common,
        evaluate_trained_model_common,
    )


def analyze_environment():
    """Analyze the ALPACA environment structure"""
    print("=" * 80)
    print("ALPACA ENVIRONMENT ANALYSIS")
    print("=" * 80)
    env = create_alpaca_env(reward_metric='ADNI_MEM')
    print_env_overview(env, title="ALPACA ENVIRONMENT ANALYSIS", target_feature='ADNI_MEM')
    return env.observation_space, env.action_space


def train_ppo_agent(total_timesteps=50000, learning_rate=3e-4, n_steps=2048, batch_size=64):
    """Train PPO agent on ALPACA environment"""
    
    print("=" * 80)
    print("PPO TRAINING ON ALPACA ENVIRONMENT")
    print("=" * 80)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"ppo_alpaca_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create tensorboard directory
    tensorboard_dir = f"tensorboard/ppo_{timestamp}"
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    print(f"Training Configuration:")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  N Steps: {n_steps}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Model Directory: {model_dir}")
    
    # Create vectorized environment
    def make_env():
        return create_alpaca_env(reward_metric='ADNI_MEM')

    vec_env = make_vec_envs(make_env, n_envs=4, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)
    eval_env = make_vec_envs(make_env, n_envs=1, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_dir,
        policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
        verbose=1,
        seed=42
    )
    
    # Set up logging
    configure_model_logger(model, f"{model_dir}/logs/")
    
    # Create callbacks
    monitor_callback = TrainingMonitorCallback(verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model/",
        log_path=f"{model_dir}/eval_logs/",
        eval_freq=max(1000, n_steps),
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=10
    )
    
    callback_list = CallbackList([monitor_callback, eval_callback])
    
    print(f"\nStarting PPO training...")
    print(f"Monitor training progress with: tensorboard --logdir {tensorboard_dir}")
    
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
        
        print(f"\nTraining completed successfully!")
        print(f"Final model saved to: {model_dir}/final_model")
        
        # Training summary
        if len(monitor_callback.episode_rewards) > 0:
            print(f"\nTraining Summary:")
            print(f"  Total Episodes: {monitor_callback.episode_count}")
            print(f"  Best Mean Reward: {monitor_callback.best_mean_reward:.4f}")
            print(f"  Final Mean Reward (last 100): {np.mean(monitor_callback.episode_rewards[-100:]):.4f}")
            print(f"  Mean Episode Length: {np.mean(monitor_callback.episode_lengths):.2f}")
        
        return model, vec_env, model_dir, monitor_callback, timestamp
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


def evaluate_random_policy(n_episodes=50):
    print("=" * 80)
    print("EVALUATING RANDOM POLICY BASELINE")
    print("=" * 80)
    def env_fn():
        return create_alpaca_env(reward_metric='ADNI_MEM')
    return evaluate_random_policy_common(env_fn, n_episodes, target_feature='ADNI_MEM')


def evaluate_trained_model(model, vec_env, random_baseline, n_episodes=20):
    print("=" * 80)
    print("EVALUATING TRAINED PPO MODEL")
    print("=" * 80)
    def env_fn():
        return create_alpaca_env(reward_metric='ADNI_MEM')
    episode_rewards, medical_outcomes, stats, _ = evaluate_trained_model_common(
        model, vec_env, env_fn, random_baseline, n_episodes, target_feature='ADNI_MEM', collect_action_stats=False
    )

    mean_reward = stats['mean_reward']
    std_reward = stats['std_reward']
    mean_adni_mem_change = np.mean([o['adni_mem_change'] for o in medical_outcomes]) if medical_outcomes else 0.0

    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Mean ADNI_MEM Change: {mean_adni_mem_change:.4f}")
    print(f"  Reward Range: [{min(episode_rewards):.4f}, {max(episode_rewards):.4f}]")

    print(f"\nComparison vs Random Baseline:")
    print(f"  Random Policy Mean: {random_baseline['mean_reward']:.4f} ± {random_baseline['std_reward']:.4f}")
    print(f"  Trained Model Mean: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Absolute Improvement: {stats['improvement']:+.4f}")
    print(f"  Percent Improvement: {stats['percent_improvement']:+.2f}%")
    print(f"  Statistical Significance: {'Yes' if stats['significant'] else 'No'} (p={stats['p_value']:.4f})")

    print(f"  Performance Assessment: {stats['performance_level']}")
    print(f"  Summary: {stats['assessment']}")
    return episode_rewards, medical_outcomes, stats


def main():
    """Main benchmark function"""
    print("PPO Benchmark for ALPACA Environment")
    print("=" * 80)
    
    try:
        # Step 1: Analyze environment
        obs_space, action_space = analyze_environment()
        
        # Step 2: Establish random policy baseline
        random_baseline = evaluate_random_policy(n_episodes=50)
        
        # Step 3: Train PPO agent
        model, vec_env, model_dir, monitor_callback, timestamp = train_ppo_agent(
            total_timesteps=500000,  # Increase for better results
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64
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
        
        print(f"\nPPO Benchmark completed successfully!")
        print(f"Results saved to: {model_dir}/")
        print(f"View training progress: tensorboard --logdir ./tensorboard")
        
        # Final assessment based on comparison
        performance_level = comparison_stats['performance_level']
        assessment = comparison_stats['assessment']
        improvement = comparison_stats['percent_improvement']
        significant = comparison_stats['significant']
        
        if performance_level == "EXCELLENT":
            print(f"EXCELLENT: {assessment}")
            if significant:
                print(f"   Statistically significant improvement!")
        elif performance_level == "GOOD":
            print(f"GOOD: {assessment}")
            if significant:
                print(f"   Statistically significant improvement!")
        elif performance_level == "MODERATE":
            print(f"MODERATE: {assessment}")
            if not significant:
                print(f"   Improvement not statistically significant")
        else:
            print(f"POOR: {assessment}")
            print("   Consider: longer training, different hyperparameters, or environment modifications")
            print("   The model may not be learning effectively from this environment")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
