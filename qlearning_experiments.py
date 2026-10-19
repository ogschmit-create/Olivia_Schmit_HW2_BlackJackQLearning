
import os
import csv
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
EPISODES = 200_000
EVAL_EPISODES = 20_000
SEED = 0

BASE_ALPHA = 0.1
BASE_GAMMA = 0.99
BASE_DECAY = 0.99995

EPSILON_START = 1.0
EPSILON_END = 0.05

os.makedirs("results", exist_ok=True)

# Each experiment changes ONE parameter at a time.
experiments = {
    "Learning Rate": [
        ("alpha_0.1", 0.1, BASE_GAMMA, BASE_DECAY),
        ("alpha_0.5", 0.5, BASE_GAMMA, BASE_DECAY),
        ("alpha_0.9", 0.9, BASE_GAMMA, BASE_DECAY),
    ],
    "Discount Factor": [
        ("gamma_0.50", BASE_ALPHA, 0.50, BASE_DECAY),
        ("gamma_0.90", BASE_ALPHA, 0.90, BASE_DECAY),
        ("gamma_0.99", BASE_ALPHA, 0.99, BASE_DECAY),
    ],
    "Epsilon Decay": [
        ("decay_0.9999", BASE_ALPHA, BASE_GAMMA, 0.9999),
        ("decay_0.99995", BASE_ALPHA, BASE_GAMMA, 0.99995),
        ("decay_0.99999", BASE_ALPHA, BASE_GAMMA, 0.99999),
    ],
}


def state(obs):
    return int(obs[0]), int(obs[1]), int(obs[2])


def train_agent(alpha, gamma, decay):
    rng = np.random.default_rng(SEED)
    env = gym.make("Blackjack-v1", natural=False, sab=False)

    Q = np.zeros((32, 11, 2, 2))
    rewards = np.zeros(EPISODES)
    epsilon = EPSILON_START

    env.reset(seed=SEED)

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            s = state(obs)

            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = int(rng.integers(2))
            else:
                q_values = Q[s]
                best_actions = np.flatnonzero(
                    q_values == q_values.max()
                )
                action = int(rng.choice(best_actions))

            next_obs, reward, terminated, truncated, _ = (
                env.step(action)
            )

            done = terminated or truncated

            # Q-learning update
            if done:
                target = reward
            else:
                target = (
                    reward
                    + gamma * np.max(Q[state(next_obs)])
                )

            Q[s][action] += alpha * (
                target - Q[s][action]
            )

            total_reward += reward
            obs = next_obs

        rewards[ep] = total_reward

        epsilon = max(
            EPSILON_END,
            epsilon * decay
        )

    env.close()
    return Q, rewards


def evaluate_agent(Q):
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env.reset(seed=999)

    eval_rewards = np.zeros(EVAL_EPISODES)

    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False

        while not done:
            # Greedy policy: no exploration
            action = int(np.argmax(Q[state(obs)]))

            obs, reward, terminated, truncated, _ = (
                env.step(action)
            )
            done = terminated or truncated

        eval_rewards[ep] = reward

    env.close()

    return {
        "mean_reward": float(np.mean(eval_rewards)),
        "win_rate": float(np.mean(eval_rewards > 0)),
        "draw_rate": float(np.mean(eval_rewards == 0)),
        "loss_rate": float(np.mean(eval_rewards < 0)),
    }


def moving_average(rewards, window=5000):
    cumulative = np.cumsum(
        np.insert(rewards, 0, 0.0)
    )
    return (
        cumulative[window:] - cumulative[:-window]
    ) / window


# ============================================================
# RUN EXPERIMENTS
# ============================================================
all_results = []
trained_agents = {}
baseline_key = "alpha_0.1"

for category, configs in experiments.items():

    print(f"\nRUNNING {category.upper()} EXPERIMENT")

    plt.figure(figsize=(10, 6))

    for name, alpha, gamma, decay in configs:

        # Avoid training duplicate baseline configurations
        if name in trained_agents:
            Q, rewards = trained_agents[name]
        elif (
            alpha == BASE_ALPHA
            and gamma == BASE_GAMMA
            and decay == BASE_DECAY
            and baseline_key in trained_agents
        ):
            Q, rewards = trained_agents[baseline_key]
        else:
            print(f"Training {name}...", flush=True)
            Q, rewards = train_agent(alpha, gamma, decay)
            trained_agents[name] = (Q, rewards)

        metrics = evaluate_agent(Q)

        print(
            f"{name}: mean={metrics['mean_reward']:.4f}, "
            f"win={metrics['win_rate']:.3f}",
            flush=True
        )

        all_results.append({
            "experiment": category,
            "configuration": name,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_decay": decay,
            **metrics,
        })

        # Save each trained agent
        np.save(f"results/{name}_Q.npy", Q)
        np.save(f"results/{name}_rewards.npy", rewards)

        smoothed = moving_average(rewards)
        x = np.arange(5000, EPISODES + 1)

        plt.plot(x, smoothed, label=name)

    plt.xlabel("Training Episode")
    plt.ylabel("Mean Reward (5000-Episode Moving Average)")
    plt.title(f"Blackjack Q-Learning: {category}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = category.lower().replace(" ", "_")
    plt.savefig(
        f"results/{filename}_comparison.png",
        dpi=150
    )
    plt.close()

# ============================================================
# SAVE SUMMARY
# ============================================================
with open(
    "results/experiment_summary.csv",
    "w",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(all_results[0].keys())
    )
    writer.writeheader()
    writer.writerows(all_results)

# ============================================================
# IDENTIFY BEST AGENT
# ============================================================
best = max(
    all_results,
    key=lambda result: result["mean_reward"]
)

print("\n" + "=" * 55)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 55)

print("Best configuration:", best["configuration"])
print("Learning rate:", best["alpha"])
print("Discount factor:", best["gamma"])
print("Epsilon decay:", best["epsilon_decay"])
print("Mean reward:", best["mean_reward"])
print("Win rate:", best["win_rate"])

print("\nSaved comparison plots and experiment_summary.csv")
print("Results folder:", os.path.abspath("results"))
