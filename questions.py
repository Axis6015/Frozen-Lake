"""
Questions and analysis for the Frozen Lake RL project.
Contains functions to generate figures and analyze results.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import FrozenLake
from model_free import sarsa, q_learning
from linear import LinearWrapper, linear_sarsa, linear_q_learning
from deep_rl import FrozenLakeImageWrapper, deep_q_network_learning


def plot_learning_curves(rewards_list, labels, title, filename):
    """
    Plot learning curves for different algorithms.
    
    Args:
        rewards_list: List of reward arrays
        labels: List of labels for each curve
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 6))
    
    for rewards, label in zip(rewards_list, labels):
        plt.plot(rewards, label=label, alpha=0.7)
    
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}')
    plt.close()


def question_2_experiments():
    """
    Run experiments for Question 2: Compare different RL algorithms.
    Generates learning curves for SARSA, Q-Learning, Linear SARSA, 
    Linear Q-Learning, and DQN.
    """
    seed = 0
    
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9
    max_episodes = 4000
    
    # TODO: Run experiments and generate plots
    # - SARSA learning curve
    # - Q-Learning learning curve
    # - Linear SARSA learning curve
    # - Linear Q-Learning learning curve
    # - DQN learning curve
    
    print("Question 2 experiments completed.")


def question_3_parameter_tuning():
    """
    Run experiments for Question 3: Parameter tuning analysis.
    Analyzes the effect of different hyperparameters on performance.
    """
    # TODO: Implement parameter tuning experiments
    # - Vary learning rate
    # - Vary exploration rate
    # - Vary discount factor
    # - Generate comparison plots
    
    print("Question 3 parameter tuning completed.")


if __name__ == '__main__':
    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    question_2_experiments()
    question_3_parameter_tuning()
