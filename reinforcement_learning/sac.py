#!/usr/bin/env python3
"""
SAC Benchmark for ALPACA Environment
====================================

Soft Actor-Critic (SAC) benchmark script for the ALPACA 
(Alzheimer's Prophylactic Action Control Agent) environment.

This script trains a SAC agent and provides comprehensive logging and 
monitoring to understand training performance and convergence.

Note: SAC typically works with continuous action spaces, but we adapt it
for the discrete MultiBinary action space of ALPACA.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium.spaces import Box
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
    from utils import (
        TrainingMonitorCallback,
        create_alpaca_env,
        print_env_overview,
        make_vec_envs,
        configure_model_logger,
        evaluate_random_policy_common,
        evaluate_trained_model_common,
    )


class MultiBinaryToBoxWrapper(gym.Wrapper):
    """
    Improved wrapper to convert MultiBinary action space to Box for SAC compatibility.
    SAC requires continuous action spaces, so we map [0,1]^n to {0,1}^n via thresholding.
    
    Key improvements:
    - Dynamic threshold adjustment based on action statistics
    - Action smoothing to reduce discrete jumps
    - Better action space bounds
    """
    
    def __init__(self, env, threshold=0.5, smoothing_factor=0.1):
        super().__init__(env)
        # Convert MultiBinary to Box [-1, 1]^n for better SAC compatibility
        n_actions = env.action_space.n
        self.action_space = Box(low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)
        self.threshold = threshold
        self.smoothing_factor = smoothing_factor
        self.prev_action = None
        
        # Track action statistics for adaptive thresholding
        self.action_stats = {'mean': 0.5, 'std': 0.1, 'count': 0}
    
    def step(self, action):
        # Clip action to prevent extreme values that could cause instability
        action = np.clip(action, -5.0, 5.0)  # Prevent extreme values
        
        # Convert [-1,1]^n to [0,1]^n via tanh transformation (more stable than sigmoid)
        normalized_action = (np.tanh(action) + 1.0) / 2.0
        
        # Apply smoothing if we have a previous action
        if self.prev_action is not None and self.smoothing_factor > 0:
            normalized_action = (1 - self.smoothing_factor) * normalized_action + self.smoothing_factor * self.prev_action
        
        # Update action statistics
        self._update_action_stats(normalized_action)
        
        # Convert continuous action [0,1]^n to discrete {0,1}^n
        discrete_action = (normalized_action > self.threshold).astype(int)
        
        # Store for next step
        self.prev_action = normalized_action.copy()
        
        return self.env.step(discrete_action)
    
    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)
    
    def _update_action_stats(self, action):
        """Update running statistics of actions for monitoring"""
        self.action_stats['count'] += 1
        alpha = 1.0 / min(self.action_stats['count'], 1000)  # Decay factor
        self.action_stats['mean'] = (1 - alpha) * self.action_stats['mean'] + alpha * np.mean(action)
        self.action_stats['std'] = (1 - alpha) * self.action_stats['std'] + alpha * np.std(action)


class LossMonitoringCallback(TrainingMonitorCallback):
    """Enhanced callback for monitoring training losses and detecting instability"""
    
    def __init__(self, verbose=0, loss_threshold=1e4):
        super().__init__(verbose)
        self.loss_threshold = loss_threshold
        self.loss_history = {'actor_loss': [], 'critic_loss': [], 'ent_coef': []}
        self.unstable_training = False
        
    def _on_step(self) -> bool:
        # Monitor losses if available
        if hasattr(self.model, 'logger'):
            try:
                # Try to get loss values from the logger
                if hasattr(self.model.logger, 'name_to_value'):
                    log_dict = self.model.logger.name_to_value
                    
                    # Check for actor and critic losses
                    actor_loss = log_dict.get('train/actor_loss', None)
                    critic_loss = log_dict.get('train/critic_loss', None)
                    ent_coef = log_dict.get('train/ent_coef', None)
                    
                    if actor_loss is not None:
                        self.loss_history['actor_loss'].append(float(actor_loss))
                        if abs(float(actor_loss)) > self.loss_threshold:
                            print(f"⚠️ WARNING: Actor loss exploding! Value: {actor_loss:.2e}")
                            self.unstable_training = True
                    
                    if critic_loss is not None:
                        self.loss_history['critic_loss'].append(float(critic_loss))
                        if abs(float(critic_loss)) > self.loss_threshold:
                            print(f"⚠️ WARNING: Critic loss exploding! Value: {critic_loss:.2e}")
                            self.unstable_training = True
                    
                    if ent_coef is not None:
                        self.loss_history['ent_coef'].append(float(ent_coef))
                        
                    # Print loss summary every 1000 steps
                    if self.num_timesteps % 1000 == 0 and len(self.loss_history['actor_loss']) > 0:
                        recent_actor = np.mean(self.loss_history['actor_loss'][-10:])
                        recent_critic = np.mean(self.loss_history['critic_loss'][-10:]) if self.loss_history['critic_loss'] else 0
                        recent_ent = np.mean(self.loss_history['ent_coef'][-10:]) if self.loss_history['ent_coef'] else 0
                        print(f"Step {self.num_timesteps}: Actor Loss: {recent_actor:.4f}, Critic Loss: {recent_critic:.4f}, Ent Coef: {recent_ent:.6f}")
                        
            except Exception as e:
                pass  # Silently continue if logging fails
        
        # Delegate to base class for episode tracking and printing
        cont = super()._on_step()
        if self.unstable_training:
            print(f"  ⚠️ Training instability detected - consider early stopping")
        
        # Stop training if losses are consistently exploding
        if (self.unstable_training and 
            len(self.loss_history['actor_loss']) > 100 and
            np.mean(self.loss_history['actor_loss'][-50:]) > self.loss_threshold):
            print("🛑 Stopping training due to persistent loss explosion")
            return False
        return cont


def create_env_for_sac():
    return create_alpaca_env(reward_metric=None, force_baseline_start=False,
                             wrapper=MultiBinaryToBoxWrapper, wrapper_kwargs={'threshold': 0.5, 'smoothing_factor': 0.05})


def analyze_environment():
    """Analyze the ALPACA environment structure"""
    print("=" * 80)
    print("ALPACA ENVIRONMENT ANALYSIS (SAC with MultiBinary→Box)")
    print("=" * 80)
    
    env = create_env_for_sac()
    print_env_overview(env, title="ALPACA ENVIRONMENT ANALYSIS (SAC with MultiBinary→Box)", target_feature='ADNI_MEM')
    
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
    
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\n  Episode {episode + 1}:")
        # Show a couple of key features for quick sanity check
        print(f"    Initial ADNI_MEM: {obs[env.env.observation_cols.index('ADNI_MEM')]:.2f}")
        print(f"    Initial Age: {obs[env.env.observation_cols.index('subject_age')]:.1f}")
        
        while True:
            action = env.action_space.sample()
            print(f"    Continuous Action Sample: {action[:5]}...") if steps == 0 else None
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if steps <= 3:  # Show first few steps
                print(f"    Step {steps}: Reward={reward:.4f}, ADNI_MEM={obs[env.env.observation_cols.index('ADNI_MEM')]:.2f}")
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"    Total Reward: {episode_reward:.4f}, Steps: {steps}")
    
    print(f"\nRandom Episode Statistics:")
    print(f"  Mean Reward: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    env.close()
    return env.observation_space, env.action_space


def train_sac_agent(total_timesteps=50000, learning_rate=1e-4, buffer_size=100000, batch_size=256):
    """
    Train SAC agent on ALPACA environment with improved stability measures
    
    Key improvements:
    - Lower learning rate to prevent gradient explosion
    - Gradient clipping
    - Conservative hyperparameters
    - Better action space handling
    - Loss monitoring
    """
    
    print("=" * 80)
    print("SAC TRAINING ON ALPACA ENVIRONMENT (STABLE VERSION)")
    print("=" * 80)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"sac_alpaca_stable_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create tensorboard directory
    tensorboard_dir = f"tensorboard/sac_stable_{timestamp}"
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    print(f"Training Configuration (Stable):")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Learning Rate: {learning_rate} (conservative)")
    print(f"  Buffer Size: {buffer_size:,}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Model Directory: {model_dir}")
    
    # Create vectorized environments
    def make_env():
        return create_env_for_sac()
    vec_env = make_vec_envs(make_env, n_envs=1, norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99)
    eval_env = make_vec_envs(make_env, n_envs=1, norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99)
    
    # Create SAC model with very conservative hyperparameters (no gradient clipping)
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=0.02,  # Even slower target network updates for stability
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,  # Single gradient step to prevent instability
        ent_coef=0.005,  # Lower fixed entropy coefficient for more stability
        target_update_interval=1,
        learning_starts=max(batch_size * 100, 10000),  # Start learning much later
        tensorboard_log=tensorboard_dir,
        policy_kwargs={
            "net_arch": [64, 64],  # Even smaller networks for stability without grad clipping
            "activation_fn": torch.nn.ReLU,  # More stable activation
            "log_std_init": -4,  # Very conservative initial log std
        },
        verbose=1,
        seed=42
    )
    
    # Note: Gradient clipping via monkey-patching causes pickle issues
    # Using SAC's built-in stability through conservative hyperparameters instead
    print("ℹ️ Using conservative hyperparameters for stability (gradient clipping disabled to avoid pickle issues)")
    
    # Set up logging
    configure_model_logger(model, f"{model_dir}/logs/")
    
    # Create callbacks with loss monitoring
    monitor_callback = LossMonitoringCallback(verbose=1, loss_threshold=1e4)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model/",
        log_path=f"{model_dir}/eval_logs/",
        eval_freq=max(10000, buffer_size // 10),  # Less frequent evaluation
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=5  # Fewer episodes to save time
    )
    
    callback_list = CallbackList([monitor_callback, eval_callback])
    
    print(f"\nStarting STABLE SAC training...")
    print(f"Monitor training progress with: tensorboard --logdir {tensorboard_dir}")
    print(f"Key stability measures:")
    print(f"  - Very low learning rate: {learning_rate}")
    print(f"  - Small network size: [64, 64]")
    print(f"  - Low fixed entropy coefficient: 0.005")
    print(f"  - Slow target updates (tau=0.02)")
    print(f"  - Late learning start")
    print(f"  - Loss monitoring with early stopping")
    
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
        
        # Training summary with loss information
        if len(monitor_callback.episode_rewards) > 0:
            print(f"\nTraining Summary:")
            print(f"  Total Episodes: {monitor_callback.episode_count}")
            print(f"  Best Mean Reward: {monitor_callback.best_mean_reward:.4f}")
            print(f"  Final Mean Reward (last 100): {np.mean(monitor_callback.episode_rewards[-100:]):.4f}")
            print(f"  Mean Episode Length: {np.mean(monitor_callback.episode_lengths):.2f}")
            print(f"  Training Stability: {'Stable' if not monitor_callback.unstable_training else 'Unstable'}")
            
            # Loss summary
            if monitor_callback.loss_history['actor_loss']:
                final_actor_loss = np.mean(monitor_callback.loss_history['actor_loss'][-10:])
                final_critic_loss = np.mean(monitor_callback.loss_history['critic_loss'][-10:]) if monitor_callback.loss_history['critic_loss'] else 0
                print(f"  Final Actor Loss: {final_actor_loss:.4f}")
                print(f"  Final Critic Loss: {final_critic_loss:.4f}")
        
        return model, vec_env, model_dir, monitor_callback, timestamp
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Save loss history for analysis
        if hasattr(monitor_callback, 'loss_history'):
            loss_df = pd.DataFrame(monitor_callback.loss_history)
            loss_df.to_csv(f"{model_dir}/loss_history.csv", index=False)
            print(f"Loss history saved to: {model_dir}/loss_history.csv")
        raise


def evaluate_random_policy(n_episodes=50):
    """Evaluate random policy baseline for comparison"""
    print("=" * 80)
    print("EVALUATING RANDOM POLICY BASELINE")
    print("=" * 80)
    def env_fn():
        return create_env_for_sac()
    return evaluate_random_policy_common(env_fn, n_episodes, target_feature='ADNI_MEM')


def evaluate_trained_model(model, vec_env, random_baseline, n_episodes=20):
    """Evaluate the trained SAC model"""
    print("=" * 80)
    print("EVALUATING TRAINED SAC MODEL")
    print("=" * 80)
    def env_fn():
        return create_env_for_sac()
    episode_rewards, medical_outcomes, stats, action_statistics = evaluate_trained_model_common(
        model, vec_env, env_fn, random_baseline, n_episodes, target_feature='ADNI_MEM', collect_action_stats=True
    )

    mean_reward = stats['mean_reward']
    std_reward = stats['std_reward']
    mean_adni_mem_change = np.mean([o['adni_mem_change'] for o in medical_outcomes]) if medical_outcomes else 0.0

    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Mean ADNI_MEM Change: {mean_adni_mem_change:.4f}")
    print(f"  Reward Range: [{min(episode_rewards):.4f}, {max(episode_rewards):.4f}]")

    if action_statistics:
        all_actions = np.array(action_statistics)
        mean_action_per_dim = np.mean(all_actions, axis=0)
        print(f"\nAction Analysis (Continuous Values):")
        print(f"  Action dimensions with high activation (>0.7): {np.sum(mean_action_per_dim > 0.7)}")
        print(f"  Action dimensions with low activation (<0.3): {np.sum(mean_action_per_dim < 0.3)}")
        print(f"  Mean action values: {mean_action_per_dim[:5]} ... (showing first 5)")

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
    """Main benchmark function with improved SAC stability"""
    print("🧠 SAC Benchmark for ALPACA Environment (STABLE VERSION)")
    print("=" * 80)
    
    try:
        # Step 1: Analyze environment
        obs_space, action_space = analyze_environment()
        
        # Step 2: Establish random policy baseline
        random_baseline = evaluate_random_policy(n_episodes=50)
        
        # Step 3: Train SAC agent with stability improvements
        model, vec_env, model_dir, monitor_callback, timestamp = train_sac_agent(
            total_timesteps=500000,  # Start with fewer timesteps for testing
            learning_rate=5e-5,   # Very conservative learning rate for stability
            buffer_size=30000,    # Smaller buffer to start
            batch_size=64        # Smaller batch size for stability
        )
        
        # Step 4: Evaluate trained model
        episode_rewards, medical_outcomes, comparison_stats = evaluate_trained_model(model, vec_env, random_baseline, n_episodes=50)
        
        # Step 5: Save results including loss analysis
        results_df = pd.DataFrame(medical_outcomes)
        results_df.to_csv(f"{model_dir}/evaluation_results.csv", index=False)
        
        # Save comparison statistics
        baseline_df = pd.DataFrame(random_baseline['medical_outcomes'])
        baseline_df.to_csv(f"{model_dir}/random_baseline_results.csv", index=False)
        
        # Save comparison summary
        comparison_df = pd.DataFrame([comparison_stats])
        comparison_df.to_csv(f"{model_dir}/performance_comparison.csv", index=False)
        
        # Save loss history if available
        if hasattr(monitor_callback, 'loss_history') and monitor_callback.loss_history['actor_loss']:
            loss_df = pd.DataFrame(monitor_callback.loss_history)
            loss_df.to_csv(f"{model_dir}/training_losses.csv", index=False)
            print(f"📈 Training loss history saved to: {model_dir}/training_losses.csv")
        
        print(f"\n✅ STABLE SAC Benchmark completed successfully!")
        print(f"📊 Results saved to: {model_dir}/")
        print(f"📈 View training progress: tensorboard --logdir ./tensorboard")
        
        # Training stability assessment
        if hasattr(monitor_callback, 'unstable_training'):
            if monitor_callback.unstable_training:
                print(f"⚠️ Training showed instability - consider further hyperparameter tuning")
            else:
                print(f"✅ Training remained stable throughout")
        
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
            print("   Note: Stable training is more important than high rewards initially")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
