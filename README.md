# Frozen Lake Reinforcement Learning Project

This project implements various reinforcement learning algorithms to solve the Frozen Lake problem.

## Project Structure

```
group[X]/
│
├── 📄 main.py                      # Main execution script
├── 📄 environment.py               # Environment implementation
├── 📄 model_based.py               # Model-based algorithms
├── 📄 model_free.py                # Model-free algorithms (SARSA, Q-Learning)
├── 📄 linear.py                    # Linear function approximation
├── 📄 deep_rl.py                   # Deep Q-Network implementation
├── 📄 questions.py                 # Experimental analysis
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # This file
│
├── 📄 output.txt                   # Generated output
├── 📄 p.npy                        # Provided data file
│
├── 📁 figures/                     # Generated figures
│   ├── q2_sarsa_curve.png
│   ├── q2_qlearning_curve.png
│   ├── q2_linear_sarsa_curve.png
│   ├── q2_linear_qlearning_curve.png
│   ├── q2_dqn_curve.png
│   └── q3_parameter_tuning.png
│
└── 📁 part2/                       # Part 2 experiments
    ├── 📄 experiments.py
    ├── 📄 analysis.py
    ├── 📄 README.md
    └── 📁 results/
        ├── training_logs.csv
        └── plots/
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Experiments

Execute the main script to run all algorithms:

```bash
python main.py
```

This will generate `output.txt` with the results of all algorithms.

### Generating Analysis Figures

Run the questions script to generate learning curves and analysis:

```bash
python questions.py
```

This will create figures in the `figures/` directory.

## Algorithms Implemented

### Model-Based Methods
- **Policy Iteration**: Iteratively evaluates and improves policies
- **Value Iteration**: Directly computes optimal value function

### Model-Free Methods
- **SARSA**: On-policy temporal difference learning
- **Q-Learning**: Off-policy temporal difference learning

### Function Approximation
- **Linear SARSA**: SARSA with linear function approximation
- **Linear Q-Learning**: Q-Learning with linear function approximation

### Deep Learning
- **Deep Q-Network (DQN)**: Deep learning with experience replay

## Environment

The Frozen Lake environment consists of:
- **Start position** (&): Where the agent begins
- **Frozen tiles** (.): Safe to walk on
- **Holes** (#): Terminal states with negative reward
- **Goal** ($): Terminal state with positive reward

The agent can slip with a certain probability, making the environment stochastic.

## Part 2: Extended Analysis

See `part2/README.md` for details on extended experiments and analysis.

## Authors

Group Y

## License

Academic project - Not for redistribution
