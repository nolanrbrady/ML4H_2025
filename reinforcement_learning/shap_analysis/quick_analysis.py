#!/usr/bin/env python3
"""
Quick SHAP Analysis Runner
==========================

Simple interface for common SHAP analysis tasks for paper writing.

Usage Examples:
    python quick_analysis.py                    # Analyze PPO (default)
    python quick_analysis.py --algorithm a2c    # Analyze A2C
    python quick_analysis.py --algorithm sac    # Analyze SAC
    python quick_analysis.py --compare          # Compare all algorithms
    python quick_analysis.py --fast             # Quick analysis (fewer samples)
    python quick_analysis.py --detailed         # Detailed analysis (more samples)
"""

import argparse
import subprocess
import sys
import os

def run_analysis(args):
    """Run the analysis with specified parameters"""
    
    cmd = [sys.executable, "unified_shap_analysis.py"]
    
    # Add algorithm
    if args.compare:
        cmd.extend(["--algorithm", "all", "--compare"])
    else:
        cmd.extend(["--algorithm", args.algorithm])
    
    # Add sample sizes based on mode
    if args.fast:
        cmd.extend(["--n_samples", "200", "--n_background", "25"])
        print("Running FAST analysis (200 samples, ~2-3 minutes)")
    elif args.detailed:
        cmd.extend(["--n_samples", "2000", "--n_background", "200"])
        print("Running DETAILED analysis (2000 samples, ~20-30 minutes)")
    else:
        cmd.extend(["--n_samples", "500", "--n_background", "50"])
        print("Running STANDARD analysis (500 samples, ~5-10 minutes)")
    
    # Add custom parameters
    if args.n_samples:
        cmd[-3] = str(args.n_samples)  # Override n_samples
    if args.n_background:
        cmd[-1] = str(args.n_background)  # Override n_background
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the analysis
    return subprocess.run(cmd)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Quick SHAP Analysis for Paper Writing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_analysis.py                    # Standard PPO analysis
  python quick_analysis.py --algorithm a2c    # Analyze A2C model
  python quick_analysis.py --compare          # Compare all algorithms
  python quick_analysis.py --fast             # Quick test (200 samples)
  python quick_analysis.py --detailed         # Thorough analysis (2000 samples)
  
Output:
  - Individual algorithm results in: {algorithm}_shap_analysis_YYYYMMDD_HHMMSS/
  - Comparative results in: comparative_analysis_YYYYMMDD_HHMMSS/
        """
    )
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'a2c', 'sac'], 
                       default='ppo', help='Algorithm to analyze (default: ppo)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all available algorithms')
    
    # Analysis modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--fast', action='store_true',
                           help='Fast analysis (200 samples, ~2-3 min)')
    mode_group.add_argument('--detailed', action='store_true',
                           help='Detailed analysis (2000 samples, ~20-30 min)')
    
    # Custom parameters (override modes)
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Custom number of samples (overrides modes)')
    parser.add_argument('--n_background', type=int, default=None,
                       help='Custom number of background samples (overrides modes)')
    
    args = parser.parse_args()
    
    print("Quick SHAP Analysis for Paper Writing")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('unified_shap_analysis.py'):
        print("Error: Must run from shap_analysis/ directory")
        print("   cd reinforcement_learning/shap_analysis/")
        return 1
    
    # Run the analysis
    try:
        result = run_analysis(args)
        
        if result.returncode == 0:
            print("\nAnalysis completed successfully!")
            print("\nCheck the generated directories for:")
            print("   • Raw SHAP data (*.npy files)")
            print("   • Individual visualizations (*.png files)")
            print("   • Summary statistics (*.csv files)")
            
            if args.compare:
                print("\nFor paper writing, examine:")
                print("   • comparative_analysis_*/feature_importance_comparison.png")
                print("   • comparative_analysis_*/treatment_comparison.png")
                print("   • comparative_analysis_*/comparative_summary.csv")
            else:
                print(f"\nFor paper writing, examine:")
                print(f"   • {args.algorithm}_shap_analysis_*/ (raw data)")
                print(f"   • {args.algorithm}_shap_visualizations_*/ (figures)")
                print(f"   • Run 'python ../shap_summary.py -m {args.algorithm}' for quick insights")
            
            return 0
        else:
            print(f"\nAnalysis failed with exit code {result.returncode}")
            return result.returncode
            
    except KeyboardInterrupt:
        print(f"\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError running analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
