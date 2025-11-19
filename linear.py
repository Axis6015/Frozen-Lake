"""
Linear function approximation methods for reinforcement learning.
Includes Linear SARSA and Linear Q-Learning.
"""

import numpy as np


class LinearWrapper:
    """Wrapper for linear function approximation."""
    
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        """Encode state into feature representation."""
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        """Decode policy from parameters."""
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
    Linear SARSA algorithm.
    
    Args:
        env: LinearWrapper environment
        max_episodes: Maximum number of episodes
        eta: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        seed: Random seed (optional)
    
    Returns:
        theta: Learned parameters
    """
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO: Implement Linear SARSA
    
    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
    Linear Q-Learning algorithm.
    
    Args:
        env: LinearWrapper environment
        max_episodes: Maximum number of episodes
        eta: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        seed: Random seed (optional)
    
    Returns:
        theta: Learned parameters
    """
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO: Implement Linear Q-Learning

    return theta
