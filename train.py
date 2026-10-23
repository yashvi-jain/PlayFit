import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from environment import PlayFitEnv
from dqn_agent import DQNAgent

def moving_average(values, window=50):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")

def main():
    os.makedirs("artifacts", exist_ok=True)

    # --- Change: use multiple environment seeds across training ---
    # Previously a single PlayFitEnv(seed=42) was reused for all 1200
    # episodes.  While the RNG state advances between reset() calls so
    # initial conditions do vary, *within* each episode the random
    # trajectories are auto-correlated.  Using a fresh seed every episode
    # exposes the agent to a wider range of player-skill / difficulty
    # combinations and prevents the replay buffer from being dominated by
    # trajectories that share similar stochastic patterns.
    agent = DQNAgent(seed=42)

    # --- Change: 1200 → 2000 episodes ---
    # The Q-value differences across actions are < 1 on a ~50-point scale.
    # 1200 episodes (≈ 80k gradient steps) is not enough for the network to
    # reliably separate such small signals, especially with lr=1e-4.
    # 2000 episodes (≈ 140k steps) gives Double-Dueling DQN enough time to
    # converge while epsilon decays to the same 0.05 floor (just slower).
    episodes = 2000

    rewards, lengths, dropouts, frustrations, engagements = [], [], [], [], []
    action_counts = np.zeros(5, dtype=int)

    for episode in range(episodes):
        # Fresh env each episode — reproducible via deterministic seed
        env = PlayFitEnv(seed=episode)
        state = env.reset()
        total_reward = 0.0
        last_info = {}
        for _ in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            last_info = info
            action_counts[action] += 1
            if done:
                break
        rewards.append(total_reward)
        lengths.append(env.step_count)
        dropouts.append(int(last_info.get("dropout", False)))
        frustrations.append(float(last_info.get("frustration", 0.0)))
        engagements.append(float(last_info.get("engagement", 0.0)))
        agent.decay_epsilon()

        if (episode + 1) % 200 == 0:
            recent = np.mean(rewards[-200:])
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"mean reward (last 200): {recent:.2f} | "
                f"epsilon: {agent.epsilon:.4f} | "
                f"action dist: {action_counts / max(action_counts.sum(), 1)}"
            )
            action_counts = np.zeros(5, dtype=int)

    agent.save("artifacts/playfit_dqn.pt")
    with open("artifacts/training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length", "dropout", "final_frustration", "final_engagement"])
        writer.writerows(zip(range(1, episodes + 1), rewards, lengths, dropouts, frustrations, engagements))

    # --- Training reward plot ---
    plt.figure()
    plt.plot(rewards, alpha=0.25, label="Episode reward")
    ma = moving_average(rewards)
    plt.plot(np.arange(len(ma)) + (len(rewards) - len(ma)), ma, label="Moving average")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("PlayFit DQN Training Rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/training_rewards.png", dpi=160)
    plt.close()
    print("Saved trained agent and training graph in artifacts/")

if __name__ == "__main__":
    main()
