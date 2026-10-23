import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from environment import PlayFitEnv
from dqn_agent import DQNAgent

N_EPISODES = 300
SEEDS = [10000 + i for i in range(N_EPISODES)]   # never used in training (train seeds are 0..1999)
DELTAS = [-0.15, -0.07, 0.0, 0.07, 0.15]
FIXED = {"Fixed Easy": 0.25, "Fixed Medium": 0.50, "Fixed Hard": 0.75}
POLICIES = ["DQN", "Rule-Based Adaptive", "Random", *FIXED]

def rule_based(obs):
    """Strong non-RL baseline: a hand-tuned controller that only sees what the agent sees
    (success trend, frustration, noisy boredom signal). Eases when struggling, hardens when coasting."""
    trend, frustration, boredom = obs[7], obs[5], obs[8]
    if frustration > 0.30 or trend < 0.40:
        return 0 if trend < 0.25 else 1
    if boredom > 0.12 or trend > 0.85:
        return 4 if boredom > 0.30 else 3
    return 2

def run_episode(policy, seed, agent, rng):
    env = PlayFitEnv(seed=seed)
    state = env.reset()
    if policy in FIXED:
        env.difficulty = FIXED[policy]
        state = env._observation()
    total_reward, diffs = 0.0, []
    counts = np.zeros(5, dtype=int)
    while True:
        if policy == "DQN":
            action = agent.act(state, explore=False)
        elif policy == "Rule-Based Adaptive":
            action = rule_based(state)
        elif policy == "Random":
            action = int(rng.integers(5))
        else:
            action = 2
        diffs.append(env.difficulty)
        counts[action] += 1
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            row = {
                "policy": policy, "seed": seed, "reward": total_reward, "length": env.step_count,
                "dropout": int(info["dropout"]), "frustration": info["frustration"],
                "boredom": info["boredom"], "engagement": info["engagement"],
                "skill_gain": info["skill"] - env.initial_skill, "mean_difficulty": float(np.mean(diffs)),
            }
            for k in range(5):
                row[f"action_{k}_count"] = int(counts[k])
            return row

def ci95(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))

def main():
    os.makedirs("artifacts", exist_ok=True)
    agent = DQNAgent(seed=42)
    agent.load("artifacts/playfit_dqn.pt")
    rng = np.random.default_rng(0)

    results = [run_episode(p, s, agent, rng) for s in SEEDS for p in POLICIES]
    with open("artifacts/episode_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    by = {p: [r for r in results if r["policy"] == p] for p in POLICIES}
    metrics = ["reward", "length", "dropout", "frustration", "boredom", "engagement", "skill_gain"]
    summary = []
    for p in POLICIES:
        row = {"policy": p}
        for m in metrics:
            vals = [r[m] for r in by[p]]
            row[f"mean_{m}"] = float(np.mean(vals))
            row[f"ci95_{m}"] = float(ci95(vals))
        summary.append(row)
    with open("artifacts/evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)

    print(f"{'policy':<22}{'reward':>16}{'length':>14}{'skill_gain':>14}{'boredom':>10}{'frustr':>9}")
    for r in summary:
        print(f"{r['policy']:<22}{r['mean_reward']:>9.2f}±{r['ci95_reward']:<5.2f}"
              f"{r['mean_length']:>9.1f}±{r['ci95_length']:<4.1f}"
              f"{r['mean_skill_gain']:>9.3f}±{r['ci95_skill_gain']:<4.3f}"
              f"{r['mean_boredom']:>8.2f}{r['mean_frustration']:>9.2f}")

    # Paired comparison: same player scenario (seed) for every policy
    print("\nPaired difference DQN - baseline (mean ± 95% CI, same 300 player scenarios):")
    dqn = {r["seed"]: r for r in by["DQN"]}
    for p in POLICIES[1:]:
        base = {r["seed"]: r for r in by[p]}
        line = f"  vs {p:<20}"
        for m in ["reward", "length", "skill_gain"]:
            d = [dqn[s][m] - base[s][m] for s in SEEDS]
            line += f" {m}: {np.mean(d):+.2f}±{ci95(d):.2f}"
        wins = np.mean([dqn[s]["reward"] > base[s]["reward"] for s in SEEDS])
        print(line + f"  | DQN wins on {wins:.0%} of scenarios")

    print("\nDQN diagnostics")
    print("Mean difficulty:", round(np.mean([x["mean_difficulty"] for x in by["DQN"]]), 3))
    tot = sum(sum(x[f"action_{k}_count"] for x in by["DQN"]) for k in range(5))
    for k in range(5):
        c = sum(x[f"action_{k}_count"] for x in by["DQN"])
        print(f"  Action {k} (delta={DELTAS[k]:+.2f}): {c / tot:.1%}")

    titles = {"reward": "Mean Episode Reward", "length": "Mean Session Length (steps)",
              "skill_gain": "Mean Skill Gain", "boredom": "Final Boredom", "frustration": "Final Frustration",
              "engagement": "Final Engagement", "dropout": "Dropout Rate"}
    for m, t in titles.items():
        plt.figure(figsize=(9, 5))
        vals = [next(r for r in summary if r["policy"] == p)[f"mean_{m}"] for p in POLICIES]
        errs = [next(r for r in summary if r["policy"] == p)[f"ci95_{m}"] for p in POLICIES]
        colors = ["tab:red" if p == "DQN" else "tab:gray" for p in POLICIES]
        plt.bar(POLICIES, vals, yerr=errs, color=colors, capsize=4)
        plt.ylabel(t)
        plt.title(f"Policy Comparison: {t}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(f"artifacts/{m}_comparison.png", dpi=160)
        plt.close()
    print("\nEvaluation complete. Files saved in artifacts/.")


if __name__ == "__main__":
    main()