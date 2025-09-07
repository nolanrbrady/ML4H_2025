#!/usr/bin/env python3
"""
SHAP Analysis Summary Script

This script provides a quick summary of SHAP analysis results,
showing the key findings in a concise format.

Usage:
    python shap_summary.py [shap_results_directory]
    python shap_summary.py --model ppo
    python shap_summary.py -m sac
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def find_latest_shap_results(model=None):
    """Find the latest SHAP analysis results directory, optionally filtered by model"""
    # Look in current directory first, then parent directory
    patterns = ["*shap_analysis_*", "../*shap_analysis_*"]
    dirs = []
    for pattern in patterns:
        dirs.extend([d for d in glob.glob(pattern) if os.path.isdir(d)])
        if dirs:  # If we found some in current directory, use those
            break
    
    if not dirs:
        raise FileNotFoundError("No SHAP analysis results found")
    
    # Filter by model if specified
    if model:
        model = model.lower()
        filtered_dirs = []
        for d in dirs:
            # Check if model name is in directory name
            if model in d.lower():
                filtered_dirs.append(d)
        
        if not filtered_dirs:
            raise FileNotFoundError(f"No SHAP analysis results found for model '{model}'")
        
        dirs = filtered_dirs
    
    dirs.sort(reverse=True)
    return dirs[0]

def load_shap_results(results_dir):
    """Load SHAP analysis results from directory"""
    try:
        # Load data
        shap_values = np.load(os.path.join(results_dir, "shap_values.npy"))
        sample_data = np.load(os.path.join(results_dir, "sample_data.npy"))
        
        # Load names
        feature_names = pd.read_csv(os.path.join(results_dir, "feature_names.csv")).iloc[:, 0].tolist()
        action_names = pd.read_csv(os.path.join(results_dir, "action_names.csv")).iloc[:, 0].tolist()
        
        # Load readable names
        readable_features = pd.read_csv(os.path.join(results_dir, "readable_feature_names.csv"))
        readable_actions = pd.read_csv(os.path.join(results_dir, "readable_action_names.csv"))
        
        # Create dictionaries
        feature_mapping = dict(zip(readable_features['original'], readable_features['readable']))
        action_mapping = dict(zip(readable_actions['original'], readable_actions['readable']))
        
        return {
            'shap_values': shap_values,
            'sample_data': sample_data,
            'feature_names': feature_names,
            'action_names': action_names,
            'feature_mapping': feature_mapping,
            'action_mapping': action_mapping
        }
    except Exception as e:
        raise RuntimeError(f"Could not load SHAP results from {results_dir}: {e}")

def analyze_feature_importance(results):
    """Analyze overall feature importance"""
    shap_values = results['shap_values']
    feature_names = results['feature_names']
    feature_mapping = results['feature_mapping']
    
    # Calculate mean absolute SHAP values across all actions and samples
    mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'readable_name': [feature_mapping.get(f, f) for f in feature_names],
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    return importance_df

def analyze_treatment_patterns(results):
    """Analyze treatment-specific patterns"""
    shap_values = results['shap_values']
    feature_names = results['feature_names']
    action_names = results['action_names']
    feature_mapping = results['feature_mapping']
    action_mapping = results['action_mapping']
    
    treatment_patterns = {}
    
    for action_idx, action_name in enumerate(action_names):
        # Get SHAP values for this action across all samples
        action_shap = shap_values[:, :, action_idx]
        
        # Calculate both mean and mean absolute values
        action_shap_mean = np.mean(action_shap, axis=0)
        action_shap_mean_abs = np.mean(np.abs(action_shap), axis=0)
        
        # For ranking, use absolute importance
        top_important_idx = np.argsort(action_shap_mean_abs)[-5:]
        bottom_important_idx = np.argsort(action_shap_mean_abs)[:5]
        
        # For direction, use the actual mean values
        treatment_patterns[action_name] = {
            'readable_name': action_mapping.get(action_name, action_name),
            'top_positive': [
                (feature_mapping.get(feature_names[idx], feature_names[idx]), 
                 action_shap_mean[idx], action_shap_mean_abs[idx])
                for idx in reversed(top_important_idx) if action_shap_mean[idx] > 0
            ],
            'top_negative': [
                (feature_mapping.get(feature_names[idx], feature_names[idx]), 
                 action_shap_mean[idx], action_shap_mean_abs[idx])
                for idx in reversed(top_important_idx) if action_shap_mean[idx] < 0
            ],
            'most_important': [
                (feature_mapping.get(feature_names[idx], feature_names[idx]), 
                 action_shap_mean[idx], action_shap_mean_abs[idx])
                for idx in reversed(top_important_idx)
            ]
        }
    
    return treatment_patterns

def print_summary(results_dir, results):
    """Print a comprehensive summary"""
    print("SHAP ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Results from: {results_dir}")
    print(f"Analyzed {results['shap_values'].shape[0]} patient samples")
    print(f"{len(results['feature_names'])} patient features")
    print(f"{len(results['action_names'])} treatment options")
    
    # Feature importance
    print("\nTOP 10 MOST IMPORTANT FEATURES:")
    importance_df = analyze_feature_importance(results)
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row.name + 1:2d}. {row['readable_name']:<30} (Score: {row['importance']:.4f})")
    
    # Treatment patterns
    print("\nTREATMENT DECISION PATTERNS:")
    print("   How to read: Shows how feature values affect treatment likelihood")
    print("      '+' = Higher values INCREASE treatment likelihood")
    print("      '-' = Higher values DECREASE treatment likelihood") 
    print("      (Opposite effects apply for lower values)")
    print("      Impact values show the strength of the effect")
    
    # Focus on treatments that actually have meaningful SHAP values
    treatment_patterns = analyze_treatment_patterns(results)
    
    meaningful_treatments = []
    
    # Find treatments with non-zero SHAP values
    for treatment in treatment_patterns.keys():
        pattern = treatment_patterns[treatment]
        if pattern['most_important']:
            max_impact = max(abs_val for _, _, abs_val in pattern['most_important'])
            if max_impact > 0.001:  # Threshold for meaningful impact
                meaningful_treatments.append(treatment)
    
    if meaningful_treatments:
        print(f"   Found {len(meaningful_treatments)} treatments with feature-dependent decisions")
        for treatment in meaningful_treatments:
            if treatment in treatment_patterns:
                pattern = treatment_patterns[treatment]
                print(f"\n{pattern['readable_name']}:")
                
                # Show most important features regardless of direction
                print("   Most INFLUENTIAL factors:")
                for feature, mean_val, abs_val in pattern['most_important'][:3]:
                    if mean_val > 0:
                        print(f"     + Higher {feature} INCREASES treatment likelihood")
                        print(f"       Lower {feature} DECREASES treatment likelihood")
                        print(f"       (impact: {mean_val:+.4f}, |impact|: {abs_val:.4f})")
                    else:
                        print(f"     - Higher {feature} DECREASES treatment likelihood")
                        print(f"       Lower {feature} INCREASES treatment likelihood")
                        print(f"       (impact: {mean_val:+.4f}, |impact|: {abs_val:.4f})")
                    print()  # Add spacing between features
                
                # Additional context for top features if helpful
                if len(pattern['most_important']) > 3:
                    remaining = len(pattern['most_important']) - 3
                    print(f"     ... and {remaining} other features with smaller impacts")
    else:
        print("   No treatments show feature-dependent decision making")
        print("   All treatment decisions appear to be deterministic/fixed")
    
    # Medical insights
    print("\nKEY MEDICAL INSIGHTS:")
    
    # Check for cognitive measures (support newer ADNI metrics)
    top_features = importance_df.head(10)['readable_name'].tolist()
    cognitive_hits = any(
        ('ADNI_MEM' in f) or ('ADNI_EF2' in f) or ('Cognitive' in f) or ('TRABSCOR' in f)
        for f in top_features
    )
    if cognitive_hits:
        print("   Agent considers cognitive measures - medically appropriate")
    else:
        print("   Cognitive measures not in top features - may need attention")
    
    if any('Brain' in f or 'Hippocampal' in f or 'Entorhinal' in f for f in top_features):
        print("   Agent uses brain imaging data - good for disease staging")
    
    if any('Tau' in f or 'Amyloid' in f for f in top_features):
        print("   Agent considers biomarkers - aligns with diagnostic criteria")
    
    if any('Age' in f for f in top_features):
        print("   Agent considers patient age - important for treatment selection")
    
    # Check AD treatment patterns
    if 'AD Treatment_active' in treatment_patterns:
        ad_pattern = treatment_patterns['AD Treatment_active']
        max_impact = max(abs_val for _, _, abs_val in ad_pattern['most_important']) if ad_pattern['most_important'] else 0
        if max_impact > 0.001:
            print("   AD Treatment decisions based on:")
            for feature, mean_val, abs_val in ad_pattern['most_important'][:3]:
                direction = "increases" if mean_val > 0 else "decreases"
                print(f"      • {feature} ({direction} likelihood, impact: {abs_val:.4f})")
        else:
            print("   AD Treatment: Deterministic policy (no feature dependence)")
    
    # Report on overall decision-making pattern
    if meaningful_treatments:
        print(f"   Algorithm shows feature-dependent decisions for {len(meaningful_treatments)} treatments:")
        for treatment in meaningful_treatments:
            readable_name = treatment_patterns[treatment]['readable_name']
            print(f"      • {readable_name}")
    else:
        print("   Algorithm uses a highly deterministic policy")
        print("   Most treatment decisions are fixed regardless of patient features")
    
    print("\nVISUALIZATION FILES:")
    # Look for visualization directory matching the results directory
    viz_dir = results_dir.replace('shap_analysis_', 'shap_visualizations_')
    if os.path.exists(viz_dir):
        print(f"   {viz_dir}/")
        print("   Individual algorithm visualizations:")
        
        # List the generated visualization files
        viz_files = [
            ("overall_feature_importance.png", "Overall feature ranking"),
            ("action_specific_importance_heatmap.png", "Treatment-specific patterns"),
            ("top_10_features.png", "Top 10 most important features"),
            ("meaningful_treatments.png", "Feature-dependent treatment decisions"),
            ("analysis_summary.png", "Analysis summary and statistics")
        ]
        
        for filename, description in viz_files:
            filepath = os.path.join(viz_dir, filename)
            if os.path.exists(filepath):
                print(f"     • {filename} - {description}")
            else:
                print(f"     {filename} - Not found")
    else:
        # Check for old naming pattern for backwards compatibility
        old_viz_dir = results_dir.replace('shap_analysis_', 'shap_visualizations_')
        if os.path.exists(old_viz_dir):
            print(f"   File: {old_viz_dir}/")
        else:
            print("    Individual visualization directory not found")
            print(f"   Expected: {viz_dir}/")
    
    # Also check for comparative analysis
    base_dir = os.path.dirname(results_dir) if os.path.dirname(results_dir) else '.'
    comp_dirs = [d for d in os.listdir(base_dir) if d.startswith('comparative_analysis_')]
    if comp_dirs:
        latest_comp = sorted(comp_dirs)[-1]
        comp_path = os.path.join(base_dir, latest_comp)
        print(f"\n   Latest comparative analysis: {comp_path}/")
        print("   Comparative visualizations:")
        print("     • feature_importance_comparison.png - Algorithm comparison")
        print("     • top_features_heatmap.png - Top features across algorithms")
        print("     • treatment_comparison.png - Treatment-specific comparisons")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate a summary of SHAP analysis results")
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['ppo', 'sac', 'a2c'],
        help='Model type to filter results by (ppo, sac, or a2c)'
    )
    
    parser.add_argument(
        'results_dir',
        nargs='?',
        help='Specific SHAP results directory (optional)'
    )
    
    args = parser.parse_args()
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        try:
            results_dir = find_latest_shap_results(args.model)
            model_str = f" for {args.model.upper()}" if args.model else ""
            print(f"Auto-found latest results{model_str}: {results_dir}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("   Run SHAP analysis first")
            return False
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        return False
    
    try:
        results = load_shap_results(results_dir)
        print_summary(results_dir, results)
        return True
    except Exception as e:
        print(f"ERROR: Error loading results: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
