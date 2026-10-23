
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from environment import PlayFitEnv
from dqn_agent import DQNAgent

N_EPISODES = 200
SEEDS = [10000 + i for i in range(N_EPISODES)]

POLICIES = {
    "DQN": None,
    "Fixed Easy": 0.25,
    "Fixed Medium": 0.50,
    "Fixed Hard": 0.75
}


def run_episode(policy, seed, agent):
    env = PlayFitEnv(seed=seed)
    state = env.reset()

    if POLICIES[policy] is not None:
        env.difficulty = POLICIES[policy]
        state = env._observation()

    total_reward = 0.0

    difficulty_history = []
    action_counts = np.zeros(5, dtype=int)

    while True:
        if policy == "DQN":
            action = agent.act(state, explore=False)
        else:
            action = 2  # Maintain the fixed difficulty.

        difficulty_history.append(env.difficulty)
        action_counts[action] += 1
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            return {
                "policy": policy,
                "seed": seed,
                "reward": total_reward,
                "length": env.step_count,
                "dropout": int(info["dropout"]),
                "frustration": info["frustration"],
                "engagement": info["engagement"],
                "skill_gain": info["skill"] - env.initial_skill,
                "mean_difficulty": np.mean(difficulty_history),
                "action_0_count": int(action_counts[0]),
                "action_1_count": int(action_counts[1]),
                "action_2_count": int(action_counts[2]),
                "action_3_count": int(action_counts[3]),
                "action_4_count": int(action_counts[4]),
            }


def main():
    os.makedirs("artifacts", exist_ok=True)

    agent = DQNAgent(seed=42)
    agent.load("artifacts/playfit_dqn.pt")

    # Every policy is tested on the same set of scenario seeds.
    episode_results = []

    for seed in SEEDS:
        for policy in POLICIES:
            episode_results.append(run_episode(policy, seed, agent))

    # Save episode-level results for reproducibility and analysis.
    with open("artifacts/episode_results.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_results[0].keys())
        writer.writeheader()
        writer.writerows(episode_results)

    metrics = [
        "reward",
        "length",
        "dropout",
        "frustration",
        "engagement",
        "skill_gain"
    ]

    summary = []

    for policy in POLICIES:
        episodes = [
            row for row in episode_results
            if row["policy"] == policy
        ]

        result = {
            "policy": policy,
            "mean_reward": np.mean([x["reward"] for x in episodes]),
            "std_reward": np.std([x["reward"] for x in episodes]),
            "mean_episode_length": np.mean([x["length"] for x in episodes]),
            "dropout_rate": np.mean([x["dropout"] for x in episodes]),
            "final_frustration": np.mean([x["frustration"] for x in episodes]),
            "final_engagement": np.mean([x["engagement"] for x in episodes]),
            "mean_skill_gain": np.mean([x["skill_gain"] for x in episodes])
        }
        summary.append(result)

    with open("artifacts/evaluation_results.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    for result in summary:
        print(
            f"\n{result['policy']}: "
            f"Reward={result['mean_reward']:.2f} "
            f"(SD={result['std_reward']:.2f}), "
            f"Length={result['mean_episode_length']:.1f}, "
            f"Dropout={result['dropout_rate']:.1%}, "
            f"Frustration={result['final_frustration']:.3f}, "
            f"Engagement={result['final_engagement']:.3f}, "
            f"Skill Gain={result['mean_skill_gain']:.3f}"
        )

    plot_metrics = {
        "mean_reward": "Mean Episode Reward",
        "mean_episode_length": "Mean Episode Length",
        "dropout_rate": "Dropout Rate",
        "final_frustration": "Final Frustration",
        "final_engagement": "Final Engagement",
        "mean_skill_gain": "Mean Skill Gain"
    }

    for metric, title in plot_metrics.items():
        plt.figure(figsize=(8, 5))
        plt.bar(
            [x["policy"] for x in summary],
            [x[metric] for x in summary]
        )
        plt.ylabel(title)
        plt.title(f"Policy Comparison: {title}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(
            f"artifacts/{metric}_comparison.png",
            dpi=160
        )
        plt.close()

    dqn_results = [
        row for row in episode_results
        if row["policy"] == "DQN"
    ]

    print("\nDQN diagnostics")
    print("Mean difficulty:",
        round(np.mean([x["mean_difficulty"] for x in dqn_results]), 3))

    total_actions = 0
    for action in range(5):
        counts = [x[f"action_{action}_count"] for x in dqn_results]
        total_actions += sum(counts)
        print(f"Action {action}: {sum(counts)} total selections")

    print("\nAction distribution:")
    for action in range(5):
        counts = sum(x[f"action_{action}_count"] for x in dqn_results)
        print(f"  Action {action} (delta={[-0.15,-0.07,0.0,0.07,0.15][action]:+.2f}): "
              f"{counts/total_actions:.1%}")

    print("\nEvaluation complete. Files saved in artifacts/.")

if __name__ == "__main__":
    main()
