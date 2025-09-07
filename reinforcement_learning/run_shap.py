#!/usr/bin/env python3
"""
SHAP Analysis Launcher
=====================

Top-level launcher for SHAP analysis tools.
Run from the reinforcement_learning directory.

Quick Commands:
    python run_shap.py                    # Standard PPO analysis
    python run_shap.py --compare          # Compare all algorithms
    python run_shap.py --algorithm a2c    # Analyze A2C
    python run_shap.py --fast             # Quick analysis
    python run_shap.py --detailed         # Detailed analysis
"""

import os
import sys
import subprocess
import argparse

def main():
    """Launch SHAP analysis tools"""
    
    # Check if we're in the right directory
    if not os.path.exists('shap_analysis'):
        print("❌ Error: Must run from reinforcement_learning directory")
        print("   This script should be run from the directory containing shap_analysis/")
        return 1
    
    # Parse arguments and pass them through
    parser = argparse.ArgumentParser(
        description='SHAP Analysis Launcher',
        add_help=False  # Let the underlying script handle help
    )
    
    # Get all command line arguments
    args = sys.argv[1:]
    
    # Build command
    cmd = [
        sys.executable, 
        os.path.join('shap_analysis', 'quick_analysis.py')
    ] + args
    
    print("🧠 Launching SHAP Analysis...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🚀 Command: {' '.join(cmd)}")
    print()
    
    # Change to shap_analysis directory and run
    try:
        os.chdir('shap_analysis')
        result = subprocess.run([sys.executable, 'quick_analysis.py'] + args)
        return result.returncode
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 