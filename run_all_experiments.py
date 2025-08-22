#!/usr/bin/env python3
"""Convenience script to run both baseline and multitask experiments."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Run all experiments in sequence."""
    
    # Ensure we're in the right directory
    if not Path("run_multitask.py").exists() or not Path("run_baselines.py").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    print("Running complete experiment suite...")
    
    # Run baseline experiments
    baseline_success = run_command(
        ["python", "run_baselines.py", "--task", "all", "--model", "all"],
        "Baseline experiments"
    )
    
    # Run multitask experiments
    multitask_success = run_command(
        ["python", "run_multitask.py", "--task", "all", "--model", "all"],
        "Multitask experiments"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUITE SUMMARY")
    print('='*60)
    print(f"Baseline experiments: {'✓ SUCCESS' if baseline_success else '✗ FAILED'}")
    print(f"Multitask experiments: {'✓ SUCCESS' if multitask_success else '✗ FAILED'}")
    
    if baseline_success and multitask_success:
        print("\n🎉 All experiments completed successfully!")
        print("Check the 'out' directory for detailed results.")
    else:
        print("\n⚠️  Some experiments failed. Check the output above for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()