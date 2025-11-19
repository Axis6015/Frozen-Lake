"""
Model-based reinforcement learning algorithms.
Includes Policy Iteration and Value Iteration.
"""

import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    """
    Evaluate a policy using iterative policy evaluation.
    
    Args:
        env: Environment model
        policy: Policy to evaluate
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        value: State value function
    """
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO: Implement policy evaluation
    
    return value


def policy_improvement(env, value, gamma):
    """
    Improve a policy given a value function.
    
    Args:
        env: Environment model
        value: State value function
        gamma: Discount factor
    
    Returns:
        policy: Improved policy
    """
    policy = np.zeros(env.n_states, dtype=int)
    
    # TODO: Implement policy improvement
    
    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    """
    Policy Iteration algorithm.
    
    Args:
        env: Environment model
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
        policy: Initial policy (optional)
    
    Returns:
        policy: Optimal policy
        value: Optimal value function
    """
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    # TODO: Implement policy iteration
    value = np.zeros(env.n_states)
    
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    """
    Value Iteration algorithm.
    
    Args:
        env: Environment model
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
        value: Initial value function (optional)
    
    Returns:
        policy: Optimal policy
        value: Optimal value function
    """
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    # TODO: Implement value iteration
    policy = np.zeros(env.n_states, dtype=int)
    
    return policy, value
