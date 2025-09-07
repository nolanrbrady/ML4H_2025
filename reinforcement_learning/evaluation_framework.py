#!/usr/bin/env python3
"""
Shared Policy Evaluation Framework
=================================

This module centralizes the shared evaluation logic used by
reinforcement_learning/benchmark_rl_algorithms.py and
treatment_comparison/benchmark_policies.py to keep behavior DRY and in sync.

It provides:
- Lightweight policy adapters with a common `predict(obs, deterministic=True, env=None)` signature
- A PolicyEvaluator that runs deterministic and stochastic comparisons,
  analyzes results, creates visualizations, and performs brain-reserve analysis.

Logic is adapted from the original benchmark scripts without changing behavior.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


# -- Policy interface and adapters -------------------------------------------------

class PolicyAdapter:
    """Unified policy adapter interface.

    Subclasses must implement `predict(obs, deterministic=True, env=None)`
    and return a binary action array compatible with the ALPACA environment.
    """

    def __init__(self, name: str):
        self.name = name

    def predict(self, obs: np.ndarray, deterministic: bool = True, env: Any = None) -> np.ndarray:
        raise NotImplementedError


class SB3PolicyAdapter(PolicyAdapter):
    """Adapter for Stable-Baselines3 policies with optional VecNormalize.

    Parameters
    - name: policy name used in reports
    - model: SB3 model instance (e.g., PPO.load(...))
    - normalizer: Optional VecNormalize instance to normalize observations
    """

    def __init__(self, name: str, model: Any, normalizer: Any = None):
        super().__init__(name)
        self.model = model
        self.normalizer = normalizer

    def predict(self, obs: np.ndarray, deterministic: bool = True, env: Any = None) -> np.ndarray:
        if self.normalizer is not None and hasattr(self.normalizer, "normalize_obs"):
            norm_obs = self.normalizer.normalize_obs(obs.reshape(1, -1))[0]
            action, _ = self.model.predict(norm_obs, deterministic=deterministic)
        else:
            action, _ = self.model.predict(obs, deterministic=deterministic)
        return action


class NoMedicationPolicyAdapter(PolicyAdapter):
    """Baseline policy that always selects 'No Medication_active' if present."""

    def __init__(self, action_names: List[str]):
        super().__init__(name="NoMed")
        self.action_names = action_names
        self._action = np.zeros(len(action_names), dtype=int)
        try:
            idx = action_names.index("No Medication_active")
            self._action[idx] = 1
        except ValueError:
            pass

    def predict(self, obs: np.ndarray, deterministic: bool = True, env: Any = None) -> np.ndarray:
        return self._action.copy()


class ClinicianPolicyAdapter(PolicyAdapter):
    """Adapter for a provided clinician policy with top-k sampling control.

    The underlying `clinician_policy` must provide:
        predict(state: np.ndarray, env, use_topk_sampling: bool) -> np.ndarray

    Parameters
    - name: policy name (e.g., 'clinician')
    - clinician_policy: object implementing the above predict signature
    - deterministic_topk: if True, use top-k sampling when deterministic=True
    - stochastic_topk: if True, use top-k sampling when deterministic=False
    """

    def __init__(
        self,
        name: str,
        clinician_policy: Any,
        deterministic_topk: bool = False,
        stochastic_topk: bool = True,
    ):
        super().__init__(name)
        self.clinician_policy = clinician_policy
        self.det_topk = deterministic_topk
        self.stoch_topk = stochastic_topk

    def predict(self, obs: np.ndarray, deterministic: bool = True, env: Any = None) -> np.ndarray:
        use_topk = self.det_topk if deterministic else self.stoch_topk
        return self.clinician_policy.predict(obs, env, use_topk_sampling=use_topk)


# -- PolicyEvaluator ---------------------------------------------------------------

class PolicyEvaluator:
    """Shared evaluator for running episodes and analyzing results.

    This combines the core logic from the RL and policy benchmark scripts.
    """

    def __init__(self, alpaca_path: str, policies: Dict[str, PolicyAdapter]):
        # Wire up environment
        self.alpaca_path = alpaca_path
        self.policies = policies

        # Import ALPACA env lazily after sys.path update
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if alpaca_path not in sys.path:
            sys.path.append(alpaca_path)
        from alpaca_env import ALPACAEnv  # type: ignore

        # Create base env to inspect spaces and columns
        original_dir = os.getcwd()
        os.chdir(alpaca_path)
        try:
            self.base_env = ALPACAEnv(data_path=".", force_baseline_start=False)
        finally:
            os.chdir(original_dir)

        self.action_names = self.base_env.action_cols
        self.observation_names = self.base_env.observation_cols

        # Derive cognitive metric used across scripts
        if "ADNI_MEM" in self.observation_names:
            self.cognitive_metric = "ADNI_MEM"
        elif "TRABSCOR" in self.observation_names:
            self.cognitive_metric = "TRABSCOR"
        else:
            # Fallback to first continuous-looking feature
            self.cognitive_metric = self.observation_names[0]

        # Storage
        self.results: Dict[str, Dict[str, Any]] = {
            "deterministic": {
                "episodes": [],
                "trajectories": {name: [] for name in self.policies.keys()},
            },
            "stochastic": {
                "episodes": [],
                "trajectories": {name: [] for name in self.policies.keys()},
            },
        }

    # -- Core episode runner --
    def run_episode(
        self,
        env: Any,
        policy_name: str,
        initial_state: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        if policy_name not in self.policies:
            raise ValueError(f"Policy {policy_name} not registered")

        # Reset env
        if initial_state is not None:
            env.state = initial_state.copy()
            env.current_step = 0
            env.done = False
            env.reward = 0
            env.info = {}
            obs = initial_state.copy()
        else:
            obs, _ = env.reset()

        episode_data: Dict[str, Any] = {
            "observations": [obs.copy()],
            "actions": [],
            "rewards": [],
            "cognitive_metric_values": [
                obs[self.observation_names.index(self.cognitive_metric)]
                if self.cognitive_metric in self.observation_names
                else np.nan
            ],
            "info": [],
            "policy": policy_name,
            "deterministic": deterministic,
            "termination_reason": None,
            "constraint_violations": [],
            "out_of_bounds_events": [],
            "uncertainty_info": [],
        }

        policy = self.policies[policy_name]
        for step in range(env.max_episode_length):
            action = policy.predict(obs, deterministic=deterministic, env=env)
            next_obs, reward, done, truncated, info = env.step(action)

            episode_data["actions"].append(action.copy())
            episode_data["rewards"].append(reward)
            episode_data["observations"].append(next_obs.copy())
            if self.cognitive_metric in self.observation_names:
                episode_data["cognitive_metric_values"].append(
                    next_obs[self.observation_names.index(self.cognitive_metric)]
                )
            else:
                episode_data["cognitive_metric_values"].append(np.nan)
            episode_data["info"].append(info)

            if info:
                if "constraint_violation" in info:
                    episode_data["constraint_violations"].append(
                        {"step": step, "violation_type": info["constraint_violation"], "action": action.copy()}
                    )
                if "out_of_bounds_variables" in info:
                    episode_data["out_of_bounds_events"].append(
                        {"step": step, "variables": info["out_of_bounds_variables"]}
                    )
                if "termination_reason" in info:
                    episode_data["termination_reason"] = info["termination_reason"]
                # Optional uncertainty fields recorded by env
                uncertainty_data = {}
                unc_key = f"{self.cognitive_metric.lower()}_uncertainty"
                for key in [unc_key, "mean_cont_uncertainty", "mean_bin_uncertainty"]:
                    if key in info:
                        uncertainty_data[key] = info[key]
                if uncertainty_data:
                    episode_data["uncertainty_info"].append({"step": step, **uncertainty_data})

            obs = next_obs
            if done or truncated:
                if done and not episode_data["termination_reason"]:
                    episode_data["termination_reason"] = "max_episode_length" if truncated else "early_termination"
                break

        # Summaries
        episode_data["total_reward"] = float(np.sum(episode_data["rewards"]))
        episode_data["final_cognitive_metric"] = float(episode_data["cognitive_metric_values"][-1])
        episode_data["cognitive_metric_change"] = float(
            episode_data["cognitive_metric_values"][-1] - episode_data["cognitive_metric_values"][0]
        )
        episode_data["episode_length"] = len(episode_data["actions"])
        return episode_data

    # -- Comparison driver --
    def run_comparison(
        self,
        num_episodes: int = 100,
        seed: int = 42,
        *,
        do_deterministic: bool = True,
        do_stochastic: bool = True,
    ) -> None:
        np.random.seed(seed)
        # Generate initial states from ALPACA for fair comparison
        initial_states: List[np.ndarray] = []
        original_dir = os.getcwd()
        os.chdir(self.alpaca_path)
        try:
            from alpaca_env import ALPACAEnv  # type: ignore
            tmp_env = ALPACAEnv(data_path=".", force_baseline_start=False)
            for _ in range(num_episodes):
                obs, _ = tmp_env.reset()
                initial_states.append(obs.copy())
        finally:
            os.chdir(original_dir)

        # Deterministic pass
        if do_deterministic:
            for idx in range(num_episodes):
                initial_state = initial_states[idx]
                self.results["deterministic"]["episodes"].append(idx)

                os.chdir(self.alpaca_path)
                try:
                    for name in self.policies.keys():
                        env = ALPACAEnv(data_path=".", force_baseline_start=False)  # type: ignore
                        traj = self.run_episode(env, name, initial_state, deterministic=True)
                        self.results["deterministic"]["trajectories"][name].append(traj)
                finally:
                    os.chdir(original_dir)

        # Stochastic pass
        if do_stochastic:
            for idx in range(num_episodes):
                initial_state = initial_states[idx]
                self.results["stochastic"]["episodes"].append(idx)

                os.chdir(self.alpaca_path)
                try:
                    for name in self.policies.keys():
                        env = ALPACAEnv(data_path=".", force_baseline_start=False)  # type: ignore
                        traj = self.run_episode(env, name, initial_state, deterministic=False)
                        self.results["stochastic"]["trajectories"][name].append(traj)
                finally:
                    os.chdir(original_dir)

    # -- Analysis --
    def analyze_results(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        if not any(self.results[ct]["episodes"] for ct in ["deterministic", "stochastic"]):
            raise ValueError("No results to analyze. Run comparison first.")

        analysis: Dict[str, Any] = {}
        for comparison_type in ["deterministic", "stochastic"]:
            if not self.results[comparison_type]["episodes"]:
                continue
            analysis[comparison_type] = self._analyze_comparison_type(comparison_type)

        # Visualizations and saving
        if output_dir:
            self._create_visualizations(analysis, output_dir)
            self._save_detailed_results(analysis, output_dir)

        # Add brain reserve analysis
        brain_reserve = self.analyze_brain_reserve_patterns(output_dir)
        analysis["brain_reserve_analysis"] = brain_reserve
        return analysis

    def _analyze_comparison_type(self, comparison_type: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # Collect per-policy metrics
        policy_metrics: Dict[str, Dict[str, List[float]]] = {}
        for name in self.policies.keys():
            episodes = self.results[comparison_type]["trajectories"][name]
            if not episodes:
                continue
            policy_metrics[name] = {
                "rewards": [ep["total_reward"] for ep in episodes],
                "final_cognitive_metrics": [ep["final_cognitive_metric"] for ep in episodes],
                "cognitive_metric_changes": [ep["cognitive_metric_change"] for ep in episodes],
                "episode_lengths": [ep["episode_length"] for ep in episodes],
                "early_terminations": [ep for ep in episodes if ep["episode_length"] < self.base_env.max_episode_length],
                "constraint_violations": sum(len(ep.get("constraint_violations", [])) for ep in episodes),
                "out_of_bounds_events": sum(len(ep.get("out_of_bounds_events", [])) for ep in episodes),
            }

        # Summary
        results["summary"] = {}
        for name, metrics in policy_metrics.items():
            results["summary"][name] = {
                "num_episodes": len(metrics["rewards"]),
                "mean_reward": float(np.mean(metrics["rewards"])) if metrics["rewards"] else 0.0,
                "std_reward": float(np.std(metrics["rewards"])) if metrics["rewards"] else 0.0,
                "mean_final_cognitive_metric": float(np.mean(metrics["final_cognitive_metrics"])) if metrics["final_cognitive_metrics"] else 0.0,
                "std_final_cognitive_metric": float(np.std(metrics["final_cognitive_metrics"])) if metrics["final_cognitive_metrics"] else 0.0,
                "mean_cognitive_metric_change": float(np.mean(metrics["cognitive_metric_changes"])) if metrics["cognitive_metric_changes"] else 0.0,
                "std_cognitive_metric_change": float(np.std(metrics["cognitive_metric_changes"])) if metrics["cognitive_metric_changes"] else 0.0,
                "mean_episode_length": float(np.mean(metrics["episode_lengths"])) if metrics["episode_lengths"] else 0.0,
                "std_episode_length": float(np.std(metrics["episode_lengths"])) if metrics["episode_lengths"] else 0.0,
                "early_termination_rate": len(metrics["early_terminations"]) / max(1, len(metrics["rewards"])),
                "constraint_violations": metrics["constraint_violations"],
                "out_of_bounds_events": metrics["out_of_bounds_events"],
            }

        # Time series of cognitive metric per step (mean ± std)
        names_list = list(self.policies.keys())
        ts_means: Dict[str, np.ndarray] = {}
        ts_stds: Dict[str, np.ndarray] = {}
        max_len = 0
        per_name_series: Dict[str, List[List[float]]] = {}
        for name in names_list:
            eps = self.results[comparison_type]["trajectories"].get(name, [])
            series = [ep.get("cognitive_metric_values", []) for ep in eps]
            per_name_series[name] = series
            if series:
                max_len = max(max_len, max(len(s) for s in series))
        if max_len > 0:
            for name in names_list:
                series = per_name_series.get(name, [])
                if not series:
                    continue
                arr = np.full((len(series), max_len), np.nan, dtype=float)
                for i, s in enumerate(series):
                    arr[i, : len(s)] = s
                ts_means[name] = np.nanmean(arr, axis=0)
                ts_stds[name] = np.nanstd(arr, axis=0)
            results["time_series"] = {"names": names_list, "means": ts_means, "stds": ts_stds}

        # Pairwise statistical tests (paired t-test for related samples)
        results["statistical_tests"] = {}
        names = list(policy_metrics.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1 :]:
                key = f"{n1}_vs_{n2}"
                r1, r2 = policy_metrics[n1]["rewards"], policy_metrics[n2]["rewards"]
                if len(r1) > 1 and len(r2) > 1 and len(r1) == len(r2):
                    # Paired t-test on per-episode total rewards
                    t_r = stats.ttest_rel(r1, r2)
                    results["statistical_tests"].setdefault(key, {})["reward_ttest"] = {
                        "statistic": float(t_r.statistic),
                        "pvalue": float(t_r.pvalue),
                        "significant": bool(t_r.pvalue < 0.05),
                        "mean_diff": float(np.mean(r1) - np.mean(r2)),
                    }
                    # Optional nonparametric paired test as a robustness check
                    try:
                        w_r = stats.wilcoxon(r1, r2, zero_method="wilcox", correction=False)
                        results["statistical_tests"][key]["reward_wilcoxon"] = {
                            "statistic": float(w_r.statistic),
                            "pvalue": float(w_r.pvalue),
                            "significant": bool(w_r.pvalue < 0.05),
                        }
                    except Exception:
                        pass

                    # Paired t-test on final cognitive metric per episode
                    c1 = policy_metrics[n1]["final_cognitive_metrics"]
                    c2 = policy_metrics[n2]["final_cognitive_metrics"]
                    t_t = stats.ttest_rel(c1, c2)
                    results["statistical_tests"][key]["cognitive_metric_ttest"] = {
                        "statistic": float(t_t.statistic),
                        "pvalue": float(t_t.pvalue),
                        "significant": bool(t_t.pvalue < 0.05),
                        "mean_diff": float(np.mean(c1) - np.mean(c2)) if c1 and c2 else 0.0,
                    }
                    try:
                        w_t = stats.wilcoxon(c1, c2, zero_method="wilcox", correction=False)
                        results["statistical_tests"][key]["cognitive_metric_wilcoxon"] = {
                            "statistic": float(w_t.statistic),
                            "pvalue": float(w_t.pvalue),
                            "significant": bool(w_t.pvalue < 0.05),
                        }
                    except Exception:
                        pass

        # Action frequency analysis (as in RL script)
        results["action_analysis"] = {}
        for name in self.policies.keys():
            episodes = self.results[comparison_type]["trajectories"][name]
            if not episodes:
                continue
            all_actions = []
            for ep in episodes:
                all_actions.extend(ep["actions"])
            if not all_actions:
                continue
            freqs = np.mean(all_actions, axis=0)
            results["action_analysis"][name] = {
                "action_frequencies": dict(zip(self.action_names, freqs)),
                "total_actions_per_step": float(np.mean([np.sum(a) for a in all_actions])),
                "most_used_actions": [self.action_names[i] for i in np.argsort(freqs)[-5:][::-1]],
                "least_used_actions": [self.action_names[i] for i in np.argsort(freqs)[:5]],
            }

        # If exactly two policies named like PPO/Clinician, compute disagreement summary like treatment script
        if set(self.policies.keys()) == set(["ppo", "clinician"]):
            ppo_eps = self.results[comparison_type]["trajectories"]["ppo"]
            cli_eps = self.results[comparison_type]["trajectories"]["clinician"]
            # Compute aggregate agreement by action
            agreements: List[np.ndarray] = []
            disagreements_per_step: List[float] = []
            for pe, ce in zip(ppo_eps, cli_eps):
                min_len = min(len(pe["actions"]), len(ce["actions"]))
                p_actions = np.array(pe["actions"][:min_len])
                c_actions = np.array(ce["actions"][:min_len])
                agreements.append(np.mean((p_actions == c_actions), axis=0))
                disagreements_per_step.append(float(np.mean(np.sum(p_actions != c_actions, axis=1))))
            if agreements:
                mean_agree = np.mean(np.stack(agreements, axis=0), axis=0)
                results["pairwise_action_differences"] = {
                    "avg_disagreement_per_step": float(np.mean(disagreements_per_step)),
                    "action_agreement_by_medication": dict(zip(self.action_names, mean_agree)),
                    "most_disagreed_actions": [self.action_names[i] for i in np.argsort(mean_agree)[:5]],
                    "most_agreed_actions": [self.action_names[i] for i in np.argsort(mean_agree)[-5:]],
                }

        return results

    # -- Visualizations (adapted from RL script) --
    def _create_visualizations(self, analysis: Dict[str, Any], output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for comparison_type in ["deterministic", "stochastic"]:
            if comparison_type not in analysis:
                continue
            comp = analysis[comparison_type]
            self._create_performance_plot(comp, output_path / comparison_type, comparison_type)
            self._create_cognitive_metric_plot(comp, output_path / comparison_type, comparison_type)
            self._create_action_heatmap(comp, output_path / comparison_type, comparison_type)
            self._create_significance_plot(comp, output_path / comparison_type, comparison_type)

    def _create_performance_plot(self, comp: Dict[str, Any], outdir: Path, comparison_type: str) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        if "summary" not in comp or not comp["summary"]:
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Policy Performance Comparison - {comparison_type.title()}", fontsize=16, fontweight="bold")
        names = list(comp["summary"].keys())
        rewards_mean = [comp["summary"][n]["mean_reward"] for n in names]
        rewards_std = [comp["summary"][n]["std_reward"] for n in names]
        axes[0, 0].bar(names, rewards_mean, yerr=rewards_std, capsize=5, alpha=0.7)
        axes[0, 0].set_title("Mean Episode Reward")
        axes[0, 0].tick_params(axis="x", rotation=45)

        final_mean = [comp["summary"][n]["mean_final_cognitive_metric"] for n in names]
        final_std = [comp["summary"][n]["std_final_cognitive_metric"] for n in names]
        axes[0, 1].bar(names, final_mean, yerr=final_std, capsize=5, alpha=0.7)
        axes[0, 1].set_title(f"Final {self.cognitive_metric}")
        axes[0, 1].tick_params(axis="x", rotation=45)

        lengths_mean = [comp["summary"][n]["mean_episode_length"] for n in names]
        lengths_std = [comp["summary"][n]["std_episode_length"] for n in names]
        axes[1, 0].bar(names, lengths_mean, yerr=lengths_std, capsize=5, alpha=0.7)
        axes[1, 0].set_title("Episode Length")
        axes[1, 0].tick_params(axis="x", rotation=45)

        early_rates = [comp["summary"][n]["early_termination_rate"] for n in names]
        axes[1, 1].bar(names, early_rates, alpha=0.7)
        axes[1, 1].set_title("Early Termination Rate")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(outdir / "performance_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_cognitive_metric_plot(self, comp: Dict[str, Any], outdir: Path, comparison_type: str) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        if "summary" not in comp or not comp["summary"]:
            return
        plt.figure(figsize=(12, 6))
        # Consistent color scheme across scripts
        color_map = {
            "PPO": "#1f77b4",           # blue
            "SAC": "#ff7f0e",           # orange
            "A2C": "#2ca02c",           # green
            "NoMed": "#d62728",         # red (legacy label)
            "No Medication": "#d62728",  # red
            # Clinician left to default cycle if not specified
        }
        for name in comp.get("time_series", {}).get("names", []):
            series = comp["time_series"]["means"][name]
            stds = comp["time_series"]["stds"][name]
            x = np.arange(len(series))
            plt.plot(x, series, label=name, color=color_map.get(name))
            plt.fill_between(x, series - stds, series + stds, alpha=0.2, color=color_map.get(name))
        plt.title(f"{self.cognitive_metric} Progression - {comparison_type.title()}")
        plt.xlabel("Step")
        plt.ylabel(self.cognitive_metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "cognitive_metric_progression.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_action_heatmap(self, comp: Dict[str, Any], outdir: Path, comparison_type: str) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        if "action_analysis" not in comp or not comp["action_analysis"]:
            return
        names = list(comp["action_analysis"].keys())
        data = []
        for n in names:
            row = [comp["action_analysis"][n]["action_frequencies"].get(a, 0.0) for a in self.action_names]
            data.append(row)
        plt.figure(figsize=(max(10, len(self.action_names) * 0.5), max(4, len(names) * 0.5)))
        sns.heatmap(data, xticklabels=self.action_names, yticklabels=names, annot=False, cmap="YlGnBu")
        plt.title(f"Action Frequency Heatmap - {comparison_type.title()}")
        plt.xlabel("Action")
        plt.ylabel("Policy")
        plt.tight_layout()
        plt.savefig(outdir / "action_frequencies_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_significance_plot(self, comp: Dict[str, Any], outdir: Path, comparison_type: str) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        # Simple placeholder: tabular CSV already includes p-values; no dedicated plot here
        # Kept to mirror the RL script structure for extensibility.
        return

    # -- Saving detailed results (CSV) --
    def _save_detailed_results(self, analysis: Dict[str, Any], output_dir: str) -> None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for comparison_type in ["deterministic", "stochastic"]:
            if comparison_type not in analysis:
                continue
            comp = analysis[comparison_type]
            comp_dir = outdir / comparison_type
            comp_dir.mkdir(parents=True, exist_ok=True)

            # Summary CSV (one row per policy)
            rows = []
            for name, s in comp.get("summary", {}).items():
                rows.append({"policy": name, **s})
            if rows:
                pd.DataFrame(rows).to_csv(comp_dir / "summary_statistics.csv", index=False)

            # If pairwise PPO vs Clinician differences were computed, save agreement matrix similar to original
            if comp.get("pairwise_action_differences"):
                agree = comp["pairwise_action_differences"]["action_agreement_by_medication"]
                pd.DataFrame({"action": list(agree.keys()), "agreement": list(agree.values())}).to_csv(
                    comp_dir / "action_agreement.csv", index=False
                )

            # Save episode-level results
            ep_rows = []
            for name in self.policies.keys():
                for idx, ep in enumerate(self.results[comparison_type]["trajectories"][name]):
                    ep_rows.append(
                        {
                            "policy": name,
                            "episode": idx,
                            "total_reward": ep["total_reward"],
                            "final_cognitive_metric": ep["final_cognitive_metric"],
                            "cognitive_metric_change": ep["cognitive_metric_change"],
                            "episode_length": ep["episode_length"],
                        }
                    )
            if ep_rows:
                pd.DataFrame(ep_rows).to_csv(comp_dir / "episode_results.csv", index=False)

            # Save pairwise statistical tests
            if comp.get("statistical_tests"):
                test_rows = []
                for pair, tests in comp["statistical_tests"].items():
                    for test_name, vals in tests.items():
                        row = {"pair": pair, "test": test_name, **vals}
                        test_rows.append(row)
                if test_rows:
                    pd.DataFrame(test_rows).to_csv(comp_dir / "pairwise_tests.csv", index=False)

            # Constraint violations and out-of-bounds (union across policies)
            viol_rows = []
            oob_rows = []
            for name in self.policies.keys():
                for idx, ep in enumerate(self.results[comparison_type]["trajectories"][name]):
                    for v in ep.get("constraint_violations", []):
                        viol_rows.append({"policy": name, "episode": idx, **v})
                    for ev in ep.get("out_of_bounds_events", []):
                        oob_rows.append({"policy": name, "episode": idx, **ev})
            if viol_rows:
                pd.DataFrame(viol_rows).to_csv(comp_dir / "constraint_violations.csv", index=False)
            if oob_rows:
                pd.DataFrame(oob_rows).to_csv(comp_dir / "out_of_bounds_events.csv", index=False)

    # -- Brain reserve analysis (generalized) --
    def analyze_brain_reserve_patterns(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # Identify required indices
        try:
            wholebrain_idx = self.observation_names.index("WholeBrain")
            icv_idx = self.observation_names.index("ICV")
        except ValueError:
            print("Brain reserve analysis skipped: 'WholeBrain' or 'ICV' not found in observations")
            return results

        # Core treatment indices (subset shown in heatmaps)
        treatment_targets = [
            "AD Treatment_active",
            "Antihypertensive_active",
            "Statin_active",
            "Diabetes Medication_active",
            "Antidepressant_active",
        ]
        available_treatments = [t for t in treatment_targets if t in self.action_names]
        treatment_indices = {t: self.action_names.index(t) for t in available_treatments}

        for comparison_type in ["deterministic", "stochastic"]:
            if not self.results[comparison_type]["episodes"]:
                continue

            # Initialize counters
            data: Dict[str, Dict[str, Dict[str, int]]] = {
                name: {cat: {t: 0 for t in available_treatments} | {"total_episodes": 0, "total_steps": 0} for cat in ["high", "mid", "low"]}
                for name in self.policies.keys()
            }

            # Accumulate counts
            num_eps = len(self.results[comparison_type]["episodes"])
            for ep_idx in range(num_eps):
                # Use first policy as source for initial obs
                first_name = list(self.policies.keys())[0]
                first_ep = self.results[comparison_type]["trajectories"][first_name][ep_idx]
                initial_obs = first_ep["observations"][0]
                icv = initial_obs[icv_idx]
                if icv == 0:
                    continue
                ratio = initial_obs[wholebrain_idx] / icv
                category = "high" if ratio > 0.71 else ("mid" if ratio >= 0.67 else "low")

                for name in self.policies.keys():
                    ep = self.results[comparison_type]["trajectories"][name][ep_idx]
                    data[name][category]["total_episodes"] += 1
                    data[name][category]["total_steps"] += len(ep["actions"])
                    for act in ep["actions"]:
                        for t, t_idx in treatment_indices.items():
                            if act[t_idx] == 1:
                                data[name][category][t] += 1

            # Convert to per-step rates
            rates: Dict[str, Dict[str, Dict[str, float]]] = {name: {"high": {}, "mid": {}, "low": {}} for name in self.policies.keys()}
            for name in self.policies.keys():
                for cat in ["high", "mid", "low"]:
                    steps = data[name][cat]["total_steps"]
                    for t in available_treatments:
                        rates[name][cat][t] = (data[name][cat][t] / steps) if steps > 0 else 0.0

            results[comparison_type] = {
                "treatment_counts": data,
                "treatment_rates": rates,
                "available_treatments": available_treatments,
            }

            # Plot heatmaps (multi-policy grid)
            self._create_brain_reserve_heatmaps(rates, available_treatments, output_dir, comparison_type)
            self._print_brain_reserve_summary(data, rates, available_treatments)

        return results

    def _create_brain_reserve_heatmaps(self, rates: Dict[str, Any], treatments: List[str], output_dir: Optional[str], heatmap_type: str) -> None:
        treatment_labels = {
            "AD Treatment_active": "AD Treatment",
            "Antihypertensive_active": "Blood Pressure Med",
            "Statin_active": "Cholesterol Med (Statin)",
            "Diabetes Medication_active": "Diabetes Med",
            "Antidepressant_active": "Anti-depressants",
        }
        categories = ["High Reserve\n(>0.71)", "Mid Reserve\n(0.67-0.71)", "Low Reserve\n(<0.67)"]
        category_keys = ["high", "mid", "low"]

        algs = list(rates.keys())
        cols = min(3, len(algs))
        rows = (len(algs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
        axes = np.array(axes).reshape(-1)

        for i, name in enumerate(algs):
            ax = axes[i]
            grid = []
            for cat in category_keys:
                row = [rates[name][cat][t] for t in treatments]
                grid.append(row)
            sns.heatmap(
                grid,
                xticklabels=[treatment_labels.get(t, t) for t in treatments],
                yticklabels=categories,
                annot=True,
                fmt=".3f",
                cmap="YlOrRd",
                ax=ax,
                cbar_kws={"label": "Treatments per Step"},
            )
            ax.set_title(f"{name} ({heatmap_type.title()})\nTreatment Patterns by Brain Reserve Ratio", fontsize=13, fontweight="bold")
            ax.set_xlabel("Treatment Type")
            ax.set_ylabel("Brain Reserve Category")
            ax.tick_params(axis="x", rotation=45, labelsize=9)

        for j in range(len(algs), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            fname = f"brain_reserve_treatment_heatmap_{heatmap_type}.png"
            plt.savefig(out / fname, dpi=300, bbox_inches="tight")
        plt.close()

    def _print_brain_reserve_summary(self, counts: Dict[str, Any], rates: Dict[str, Any], treatments: List[str]) -> None:
        print("\n" + "=" * 80)
        print("BRAIN RESERVE TREATMENT PATTERN ANALYSIS - ALL POLICIES")
        print("=" * 80)
        algs = list(counts.keys())
        print("\nEpisode and Step Distribution by Brain Reserve Category:")
        for name in algs:
            print(f"\n{name}:")
            for cat, cat_name in zip(["high", "mid", "low"], ["High Reserve (>0.71)", "Mid Reserve (0.67-0.71)", "Low Reserve (<0.67)"]):
                ep = counts[name][cat]["total_episodes"]
                steps = counts[name][cat]["total_steps"]
                avg = steps / ep if ep > 0 else 0.0
                print(f"  {cat_name}: {ep} episodes, {steps} steps (avg: {avg:.1f} steps/episode)")

        print("\nTreatment Rates (treatments per step) by Brain Reserve Category:")
        for t in treatments:
            print(f"\n{t}:")
            header = "  Category          " + " ".join([f"{n:>12}" for n in algs])
            print(header)
            print("  " + "-" * (18 + 13 * len(algs)))
            for cat, cat_short in zip(["high", "mid", "low"], ["High Reserve   ", "Mid Reserve    ", "Low Reserve    "]):
                row = f"  {cat_short}   " + " ".join([f"{rates[n][cat][t]:11.3f}" for n in algs])
                print(row)
