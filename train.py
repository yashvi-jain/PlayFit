import os
import csv
import sys
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
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    tag = "" if seed == 42 else f"_seed{seed}"
    os.makedirs("artifacts", exist_ok=True)

    agent = DQNAgent(seed=seed)
    episodes = 2000 

    rewards, lengths, dropouts, frustrations, engagements = [], [], [], [], []
    action_counts = np.zeros(5, dtype=int)

    for episode in range(episodes):
        # Fresh env each episode, reproducible via deterministic seed
        env = PlayFitEnv(seed=episode)
        state = env.reset()
        total_reward = 0.0
        last_info = {}
        for _ in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            # bootstrap through time-limit truncation; only a real dropout is terminal
            agent.remember(state, action, reward, next_state, info['dropout'])
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

    agent.save(f"artifacts/playfit_dqn{tag}.pt")
    with open(f"artifacts/training_log{tag}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length", "dropout", "final_frustration", "final_engagement"])
        writer.writerows(zip(range(1, episodes + 1), rewards, lengths, dropouts, frustrations, engagements))

    # training reward plot
    plt.figure()
    plt.plot(rewards, alpha=0.25, label="Episode reward")
    ma = moving_average(rewards)
    plt.plot(np.arange(len(ma)) + (len(rewards) - len(ma)), ma, label="Moving average")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("PlayFit DQN Training Rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"artifacts/training_rewards{tag}.png", dpi=160)
    plt.close()
    print("Saved trained agent and training graph in artifacts/")

if __name__ == "__main__":
    main()