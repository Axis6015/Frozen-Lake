"""
Part 2: Extended experiments and comparative analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('..')

from environment import FrozenLake
from model_free import sarsa, q_learning
from linear import LinearWrapper, linear_sarsa, linear_q_learning
from deep_rl import FrozenLakeImageWrapper, deep_q_network_learning


class ExperimentRunner:
    """Run and log experiments for comparative analysis."""
    
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_experiment(self, algorithm_name, algorithm_func, env, params):
        """
        Run a single experiment and log results.
        
        Args:
            algorithm_name: Name of the algorithm
            algorithm_func: Function implementing the algorithm
            env: Environment to run on
            params: Dictionary of parameters
        
        Returns:
            results: Dictionary with experiment results
        """
        print(f"Running {algorithm_name}...")
        
        # TODO: Implement experiment runner
        # - Time the algorithm
        # - Track learning progress
        # - Compute final performance metrics
        # - Log results
        
        result = {
            'algorithm': algorithm_name,
            'params': params,
            'final_value': 0.0,
            'convergence_episodes': 0,
            'runtime': 0.0
        }
        
        self.results.append(result)
        return result
    
    def save_results(self, filename='training_logs.csv'):
        """Save all results to CSV."""
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / filename, index=False)
        print(f"Results saved to {self.output_dir / filename}")
    
    def generate_plots(self):
        """Generate comparison plots."""
        # TODO: Create comparative visualizations
        pass


def main():
    """Run all Part 2 experiments."""
    seed = 0
    
    # Large lake for more complex experiments
    lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', '.', '#', '.'],
            ['.', '#', '.', '.', '#', '.', '#', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '$']]
    
    env = FrozenLake(lake, slip=0.1, max_steps=64, seed=seed)
    
    runner = ExperimentRunner()
    
    # TODO: Define experiments to run
    # - Different algorithms
    # - Different hyperparameters
    # - Different environment configurations
    
    runner.save_results()
    runner.generate_plots()
    
    print("All Part 2 experiments completed!")


if __name__ == '__main__':
    main()
