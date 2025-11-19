"""
Part 2: Statistical analysis and visualization of experimental results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ResultsAnalyzer:
    """Analyze and visualize experimental results."""
    
    def __init__(self, results_file='results/training_logs.csv'):
        self.results_file = Path(results_file)
        self.df = None
        self.load_results()
    
    def load_results(self):
        """Load results from CSV file."""
        if self.results_file.exists():
            self.df = pd.read_csv(self.results_file)
            print(f"Loaded {len(self.df)} results from {self.results_file}")
        else:
            print(f"Results file {self.results_file} not found.")
    
    def summary_statistics(self):
        """Compute summary statistics for all algorithms."""
        if self.df is None:
            return
        
        print("\n=== Summary Statistics ===")
        print(self.df.groupby('algorithm').agg({
            'final_value': ['mean', 'std', 'min', 'max'],
            'convergence_episodes': ['mean', 'std'],
            'runtime': ['mean', 'std']
        }))
    
    def plot_algorithm_comparison(self):
        """Create comparison plots for different algorithms."""
        if self.df is None:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Final value comparison
        self.df.boxplot(column='final_value', by='algorithm', ax=axes[0])
        axes[0].set_title('Final Value by Algorithm')
        axes[0].set_xlabel('Algorithm')
        axes[0].set_ylabel('Final Value')
        
        # Convergence speed comparison
        self.df.boxplot(column='convergence_episodes', by='algorithm', ax=axes[1])
        axes[1].set_title('Convergence Speed')
        axes[1].set_xlabel('Algorithm')
        axes[1].set_ylabel('Episodes to Converge')
        
        # Runtime comparison
        self.df.boxplot(column='runtime', by='algorithm', ax=axes[2])
        axes[2].set_title('Runtime Comparison')
        axes[2].set_xlabel('Algorithm')
        axes[2].set_ylabel('Runtime (seconds)')
        
        plt.tight_layout()
        plt.savefig('results/plots/algorithm_comparison.png', dpi=300)
        plt.close()
        print("Saved algorithm comparison plot.")
    
    def plot_parameter_sensitivity(self, param_name):
        """Plot sensitivity to a specific parameter."""
        # TODO: Implement parameter sensitivity analysis
        pass
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n=== Generating Analysis Report ===")
        
        self.summary_statistics()
        self.plot_algorithm_comparison()
        
        # TODO: Add more analysis
        # - Statistical significance tests
        # - Parameter sensitivity analysis
        # - Learning curve comparisons
        # - Performance vs. complexity trade-offs
        
        print("\nAnalysis complete! Check results/plots/ for visualizations.")


def main():
    """Run all analysis."""
    analyzer = ResultsAnalyzer()
    analyzer.generate_report()


if __name__ == '__main__':
    main()
