"""
Model-free reinforcement learning algorithms.
Includes SARSA and Q-Learning.
"""

import numpy as np


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
    SARSA algorithm for model-free RL.
    
    Args:
        env: Environment
        max_episodes: Maximum number of episodes
        eta: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        seed: Random seed (optional)
    
    Returns:
        policy: Learned policy
        value: Learned value function
    """
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO: Implement SARSA algorithm
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
    Q-Learning algorithm for model-free RL.
    
    Args:
        env: Environment
        max_episodes: Maximum number of episodes
        eta: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        seed: Random seed (optional)
    
    Returns:
        policy: Learned policy
        value: Learned value function
    """
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO: Implement Q-Learning algorithm
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value
