#!/usr/bin/env python3
"""
Clinician Policy Fidelity Test (clean)
=====================================

Computes concise fidelity metrics for the clinician Bayesian policy
trained in clinician_bayesian_model.py. Focuses on how well the
model mimics clinician actions (multi-label):
- Exact match accuracy (subset accuracy)
- Hamming loss
- F1 scores (micro, macro, weighted)
- Per-action F1 and supports

Usage:
  python clinician_fidelity_test.py --model_path ./best_clinician_policy.pth \
    --threshold 0.5 --monte_carlo_samples 50
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    label_ranking_average_precision_score,
    coverage_error,
)

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import the exact model used for training
from clinician_bayesian_model import BayesianModel  # noqa: E402


def load_schema(schema_path: str) -> Dict[str, Any]:
    if not os.path.exists(schema_path):
        return {}
    with open(schema_path, 'r') as f:
        return json.load(f)


def align_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    return df[present]


def predict_probs(
    model: BayesianModel,
    X: pd.DataFrame,
    device: torch.device,
    mc_samples: int,
    return_stack: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X_tensor = torch.tensor(X.values, dtype=torch.float32, device=device)
    probs = []
    for _ in range(max(1, mc_samples)):
        with torch.no_grad():
            _, p = model.predict(X_tensor)
            probs.append(p.cpu().numpy())
    mc_stack = np.stack(probs, axis=0) if len(probs) > 0 else None
    mean_probs = mc_stack.mean(axis=0) if mc_stack is not None else np.zeros((X.shape[0], 0))
    return (mean_probs, mc_stack) if return_stack else (mean_probs, None)


def entropy_binary(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if y_prob.size == 0:
        return float('nan')
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        w = np.mean(mask)
        ece += w * abs(acc - conf)
    return float(ece)


def sweep_global_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    best_t = float(thresholds[0])
    best_macro = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if macro > best_macro:
            best_macro = macro
            best_t = float(t)
    return best_t, float(best_macro)


def sweep_per_action_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n_labels = y_true.shape[1]
    best = np.full(n_labels, 0.5, dtype=float)
    for i in range(n_labels):
        fbest = -1.0
        tbest = 0.5
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            best[i] = 0.5
            continue
        for t in thresholds:
            ypi = (yp >= t).astype(int)
            f1i = f1_score(yt, ypi, zero_division=0)
            if f1i > fbest:
                fbest = f1i
                tbest = float(t)
        best[i] = tbest
    return best


def selective_risk_curve(y_true: np.ndarray, y_prob: np.ndarray, sample_uncertainty: np.ndarray, steps: List[float]) -> pd.DataFrame:
    idx = np.argsort(sample_uncertainty)  # most certain first
    y_true_sorted = y_true[idx]
    y_prob_sorted = y_prob[idx]
    rows = []
    for cov in steps:
        k = max(1, int(len(y_true_sorted) * cov))
        y_pred = (y_prob_sorted[:k] >= 0.5).astype(int)
        yt = y_true_sorted[:k]
        rows.append({
            'coverage': cov,
            'micro_f1': float(f1_score(yt, y_pred, average="micro", zero_division=0)),
            'macro_f1': float(f1_score(yt, y_pred, average="macro", zero_division=0)),
            'hamming_loss': float(hamming_loss(yt, y_pred)),
            'subset_accuracy': float(accuracy_score(yt, y_pred)),
            'n_samples': int(k),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Clinician policy fidelity metrics')
    parser.add_argument('--model_path', type=str, default=os.path.join(CURRENT_DIR, 'best_clinician_policy.pth'))
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--monte_carlo_samples', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default=None, help='Optional directory to save metrics CSV/JSON')
    parser.add_argument('--val_X', type=str, default=None, help='Optional path to validation X CSV for threshold tuning')
    parser.add_argument('--val_y', type=str, default=None, help='Optional path to validation y CSV for threshold tuning')
    parser.add_argument('--ece_bins', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cpu')

    # Load data
    X_test_path = os.path.join(CURRENT_DIR, 'clinician_X_test.csv')
    y_test_path = os.path.join(CURRENT_DIR, 'clinician_y_test.csv')
    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        raise FileNotFoundError(f'Missing test data at {X_test_path} and/or {y_test_path}')
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    if 'subject_id' in X_test.columns:
        X_test = X_test.drop(columns=['subject_id'])
    if 'subject_id' in y_test.columns:
        y_test = y_test.drop(columns=['subject_id'])

    # Align to schema (mirrors training script)
    schema = load_schema(os.path.join(CURRENT_DIR, 'columns_schema.json'))
    state_cols = schema.get('clinician_state_cols') or schema.get('model_input_cols')
    # Ensure absolute time is excluded to mirror training
    if state_cols and 'months_since_bl' in state_cols:
        state_cols = [c for c in state_cols if c != 'months_since_bl']
    action_cols = schema.get('clinician_action_cols') or schema.get('action_cols')
    if state_cols:
        X_test = align_columns(X_test, state_cols)
    if action_cols:
        y_test = align_columns(y_test, action_cols)

    input_size = X_test.shape[1]
    num_actions = y_test.shape[1]

    # Build and load model
    model = BayesianModel(
        input_size=input_size,
        num_continuous_outputs=0,
        num_binary_outputs=num_actions,
    ).to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Optional validation load for threshold tuning
    X_val = None
    y_val = None
    if args.val_X and args.val_y and os.path.exists(args.val_X) and os.path.exists(args.val_y):
        X_val = pd.read_csv(args.val_X)
        y_val = pd.read_csv(args.val_y)
        if 'subject_id' in X_val.columns:
            X_val = X_val.drop(columns=['subject_id'])
        if 'subject_id' in y_val.columns:
            y_val = y_val.drop(columns=['subject_id'])
        if state_cols:
            X_val = align_columns(X_val, state_cols)
        if action_cols:
            y_val = align_columns(y_val, action_cols)

    # Predict
    y_prob, mc_stack = predict_probs(model, X_test, device, args.monte_carlo_samples, return_stack=True)
    y_pred = (y_prob >= args.threshold).astype(int)
    y_true = y_test.values.astype(int)

    # Metrics
    exact = float(np.mean(np.all(y_true == y_pred, axis=1)))
    subset_acc = accuracy_score(y_true, y_pred)
    h_loss = hamming_loss(y_true, y_pred)

    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-action metrics, AUCs, calibration
    per_action = {}
    action_names = list(y_test.columns)
    for i, name in enumerate(action_names):
        # AUCs (safe if class is constant)
        try:
            auroc = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except Exception:
            auroc = float('nan')
        try:
            aupr = float(average_precision_score(y_true[:, i], y_prob[:, i]))
        except Exception:
            aupr = float('nan')
        # Calibration
        try:
            brier = float(brier_score_loss(y_true[:, i], y_prob[:, i]))
        except Exception:
            brier = float('nan')
        ece = expected_calibration_error(y_true[:, i], y_prob[:, i], n_bins=args.ece_bins)

        per_action[name] = {
            'precision': float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            'recall': float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            'f1': float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            'auroc': auroc,
            'aupr': aupr,
            'brier': brier,
            'ece': float(ece),
            'support_true_positives': int(np.sum(y_true[:, i] == 1)),
            'support_true_negatives': int(np.sum(y_true[:, i] == 0)),
            'predicted_positives': int(np.sum(y_pred[:, i] == 1)),
            'true_prevalence': float(np.mean(y_true[:, i])),
            'predicted_prevalence': float(np.mean(y_pred[:, i])),
        }

    # Coverage (avg labels per sample)
    coverage_pred = float(np.mean(np.sum(y_pred, axis=1)))
    coverage_true = float(np.mean(np.sum(y_true, axis=1)))

    # Ranking metrics
    try:
        lrap = float(label_ranking_average_precision_score(y_true, y_prob))
    except Exception:
        lrap = float('nan')
    try:
        cov_err = float(coverage_error(y_true, y_prob))
    except Exception:
        cov_err = float('nan')

    # No-med exclusivity
    no_med_name = 'No Medication_active'
    exclusivity_conflict_rate = None
    if no_med_name in action_names:
        idx_no = action_names.index(no_med_name)
        conflict_pred = np.sum((y_pred[:, idx_no] == 1) & (np.sum(y_pred, axis=1) > 1))
        exclusivity_conflict_rate = float(conflict_pred / y_pred.shape[0])

    # Hamming distance distribution
    hamming_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)

    summary = {
        'n_samples': int(X_test.shape[0]),
        'n_actions': int(num_actions),
        'threshold': float(args.threshold),
        'monte_carlo_samples': int(args.monte_carlo_samples),
        'exact_match_accuracy': exact,
        'subset_accuracy': float(subset_acc),
        'hamming_loss': float(h_loss),
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'coverage_predicted': coverage_pred,
        'coverage_true': coverage_true,
        'hamming_mean': float(np.mean(hamming_per_sample)),
        'hamming_median': float(np.median(hamming_per_sample)),
        'hamming_p90': float(np.percentile(hamming_per_sample, 90)),
        'hamming_p95': float(np.percentile(hamming_per_sample, 95)),
        'lrap': lrap,
        'coverage_error': cov_err,
        'no_med_exclusivity_conflict_rate': exclusivity_conflict_rate,
    }

    # Calibration overall means
    brier_overall = np.nanmean([v['brier'] for v in per_action.values()])
    ece_overall = np.nanmean([v['ece'] for v in per_action.values()])
    summary['brier_mean'] = float(brier_overall)
    summary['ece_mean'] = float(ece_overall)

    # Threshold tuning (if validation provided)
    tuned = {}
    if X_val is not None and y_val is not None:
        y_val_true = y_val.values.astype(int)
        y_val_prob, _ = predict_probs(model, X_val, device, args.monte_carlo_samples, return_stack=False)
        grid = np.linspace(0.05, 0.95, 19)
        best_t, best_val_macro = sweep_global_threshold(y_val_true, y_val_prob, grid)
        tuned['global_threshold'] = best_t
        tuned['val_macro_f1_at_best'] = best_val_macro
        # Apply on test
        y_pred_best = (y_prob >= best_t).astype(int)
        tuned['test_macro_f1'] = float(f1_score(y_true, y_pred_best, average='macro', zero_division=0))
        tuned['test_micro_f1'] = float(f1_score(y_true, y_pred_best, average='micro', zero_division=0))
        # Per-action thresholds
        per_thr = sweep_per_action_thresholds(y_val_true, y_val_prob, grid)
        tuned['per_action_thresholds'] = {name: float(per_thr[i]) for i, name in enumerate(action_names)}
        # Apply per-action thresholds on test
        y_pred_pa = (y_prob >= per_thr.reshape(1, -1)).astype(int)
        tuned['test_macro_f1_per_action_thresholds'] = float(f1_score(y_true, y_pred_pa, average='macro', zero_division=0))
        tuned['test_micro_f1_per_action_thresholds'] = float(f1_score(y_true, y_pred_pa, average='micro', zero_division=0))

    # Uncertainty diagnostics (MC)
    if mc_stack is not None and mc_stack.shape[0] > 1:
        std_across_mc = mc_stack.std(axis=0)  # (n, k)
        entropy_mat = entropy_binary(y_prob)  # (n, k)
        # Sample-level uncertainty (mean across actions)
        sample_uncertainty = entropy_mat.mean(axis=1)
        # Selective risk curve
        steps = [i / 10.0 for i in range(1, 11)]
        sel_df = selective_risk_curve(y_true, y_prob, sample_uncertainty, steps)
    else:
        std_across_mc = None
        entropy_mat = entropy_binary(y_prob)
        sample_uncertainty = entropy_mat.mean(axis=1)
        sel_df = selective_risk_curve(y_true, y_prob, sample_uncertainty, [i / 10.0 for i in range(1, 11)])

    # Uncertainty summary (sample-level entropy across actions)
    summary['uncertainty_mean'] = float(np.mean(sample_uncertainty))
    summary['uncertainty_median'] = float(np.median(sample_uncertainty))
    summary['uncertainty_p90'] = float(np.percentile(sample_uncertainty, 90))
    summary['uncertainty_p95'] = float(np.percentile(sample_uncertainty, 95))
    summary['high_uncertainty_share_entropy>0.5'] = float(np.mean(sample_uncertainty > 0.5))

    # Print concise report
    print('Clinician Policy Fidelity Metrics')
    print('- Samples:', summary['n_samples'], 'Actions:', summary['n_actions'])
    print('- Exact Match:', f"{summary['exact_match_accuracy']:.3f}")
    print('- Hamming Loss:', f"{summary['hamming_loss']:.3f}")
    print('- F1 (micro/macro/weighted):',
          f"{summary['micro_f1']:.3f}/",
          f"{summary['macro_f1']:.3f}/",
          f"{summary['weighted_f1']:.3f}")
    print('- Coverage (pred/true):',
          f"{summary['coverage_predicted']:.2f}/",
          f"{summary['coverage_true']:.2f}")
    print('- Ranking metrics (LRAP, coverage error):', f"{summary['lrap']:.3f}", f"{summary['coverage_error']:.3f}")
    if summary.get('no_med_exclusivity_conflict_rate') is not None:
        print('- NoMed exclusivity conflicts:', f"{summary['no_med_exclusivity_conflict_rate']:.3f}")
    print('- Calibration (Brier/ECE mean):', f"{summary['brier_mean']:.3f}", f"{summary['ece_mean']:.3f}")

    # Optional save
    if args.save_dir:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(args.save_dir, f'clinician_fidelity_{ts}')
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'summary_metrics.csv'), index=False)
        pa_df = pd.DataFrame.from_dict(per_action, orient='index').reset_index().rename(columns={'index': 'action'})
        pa_df.to_csv(
            os.path.join(out_dir, 'per_action_metrics.csv'), index=False
        )
        pd.DataFrame(y_prob, columns=action_names).to_csv(os.path.join(out_dir, 'predicted_probs.csv'), index=False)
        # Save threshold tuning if available
        if X_val is not None and y_val is not None and len(tuned) > 0:
            pd.DataFrame([
                {
                    'global_threshold': tuned.get('global_threshold'),
                    'val_macro_f1': tuned.get('val_macro_f1_at_best'),
                    'test_macro_f1_with_global': tuned.get('test_macro_f1'),
                    'test_micro_f1_with_global': tuned.get('test_micro_f1'),
                    'test_macro_f1_with_per_action': tuned.get('test_macro_f1_per_action_thresholds'),
                    'test_micro_f1_with_per_action': tuned.get('test_micro_f1_per_action_thresholds'),
                }
            ]).to_csv(os.path.join(out_dir, 'threshold_tuning_summary.csv'), index=False)
            if 'per_action_thresholds' in tuned:
                pd.DataFrame(
                    [{'action': k, 'threshold': v} for k, v in tuned['per_action_thresholds'].items()]
                ).to_csv(os.path.join(out_dir, 'per_action_thresholds.csv'), index=False)
        # Save selective risk curve and uncertainty summaries
        sel_df.to_csv(os.path.join(out_dir, 'selective_risk_curve.csv'), index=False)
        # Uncertainty per action (entropy and, if available, std across MC)
        df_unc = pd.DataFrame({'action': action_names,
                               'entropy_mean': [np.mean(entropy_mat[:, i]) for i in range(len(action_names))]})
        if std_across_mc is not None:
            df_unc['std_mean'] = [np.mean(std_across_mc[:, i]) for i in range(len(action_names))]
        df_unc.to_csv(os.path.join(out_dir, 'uncertainty_per_action.csv'), index=False)
        print('Saved metrics to:', out_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
