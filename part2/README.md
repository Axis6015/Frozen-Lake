# Part 2: Extended Analysis

This directory contains extended experiments and comparative analysis for the Frozen Lake RL project.

## Overview

Part 2 focuses on:
1. **Comparative Analysis**: Systematic comparison of different RL algorithms
2. **Hyperparameter Tuning**: Analysis of parameter sensitivity
3. **Statistical Analysis**: Rigorous statistical evaluation of results
4. **Scalability Testing**: Performance on larger environments

## Files

- `experiments.py`: Run extended experiments with different configurations
- `analysis.py`: Statistical analysis and visualization of results
- `results/training_logs.csv`: Logged experimental data
- `results/plots/`: Generated visualizations

## Running Experiments

### Step 1: Run Experiments

```bash
cd part2
python experiments.py
```

This will:
- Run multiple trials of each algorithm
- Test different hyperparameter configurations
- Log all results to `results/training_logs.csv`

### Step 2: Analyze Results

```bash
python analysis.py
```

This will:
- Compute summary statistics
- Generate comparison plots
- Perform statistical significance tests
- Create visualizations in `results/plots/`

## Experiments

### Experiment 1: Algorithm Comparison
Compare performance of:
- SARSA
- Q-Learning
- Linear SARSA
- Linear Q-Learning
- Deep Q-Network

Metrics:
- Final policy value
- Convergence speed
- Sample efficiency
- Computational cost

### Experiment 2: Hyperparameter Sensitivity
Analyze effect of:
- Learning rate (η)
- Discount factor (γ)
- Exploration rate (ε)
- Network architecture (for DQN)

### Experiment 3: Environment Complexity
Test on:
- Different grid sizes
- Varying slip probabilities
- Different obstacle densities

## Results Structure

```
results/
├── training_logs.csv          # All experimental data
└── plots/
    ├── algorithm_comparison.png
    ├── learning_curves.png
    ├── parameter_sensitivity.png
    └── statistical_tests.png
```

## Analysis Methods

### Performance Metrics
- **Final Value**: Value of learned policy
- **Convergence Rate**: Episodes to convergence
- **Sample Efficiency**: Reward per episode
- **Stability**: Variance in performance

### Statistical Tests
- ANOVA for algorithm comparison
- T-tests for pairwise comparisons
- Confidence intervals
- Effect size analysis

## Expected Outcomes

1. **Algorithm Rankings**: Which algorithms work best for Frozen Lake
2. **Parameter Guidelines**: Optimal hyperparameter ranges
3. **Trade-offs**: Performance vs. computational cost
4. **Insights**: When to use which algorithm

## Future Work

- Transfer learning experiments
- Multi-task learning
- Meta-learning approaches
- Real-world applications

## Authors

Group Y
