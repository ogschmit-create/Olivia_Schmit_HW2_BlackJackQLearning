# Olivia_Schmit_HW2_BlackJackQLearning

This project implements a Q-learning agent for the Blackjack-v1 environment in Gymnasium. The objective is to make decisions using reinforcement learning and evaluate how changing different parameters (alpha, gamma, and epsilon-decay) impact learning performance.

## Setup

These files require Python 3.10 or newer.

```bash
pip install numpy matplotlib "gymnasium[toy-text]" imageio imageio-ffmpeg
```

Variables at the top of `qlearning_experiments.py` can be changed to test the agent under different parameters.

## Run

```bash
python qlearning_experiments.py   # trains and evaluates all agents, makes the comparison plots
python video.py                   # records a video of the best agent playing
```

## Files

- `qlearning_experiments.py`: Q-learning agent, training, evaluation, and all experiments
- `video.py`: records `blackjack_agent.mp4` using the best agent
- Results: comparison plots (`learning_rate_comparison.png`, `discount_factor_comparison.png`, `epsilon_decay_comparison.png`), `experiment_summary.csv`, trained Q-tables (`.npy`), and the video
