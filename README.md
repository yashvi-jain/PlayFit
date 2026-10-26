# PlayFit: Personalized Game Difficulty Adaptation with Reinforcement Learning

> AI Course Assignment 2: Real-Life Use Cases of Reinforcement Learning

An RL agent that observes player performance, mistakes, and behavioural feedback, then adapts enemy strength, time limits, hints, and recovery opportunities. Its reward encourages sustained engagement and skill improvement and penalizes frustration and simulated player dropout.

**Team Members**

| Name | Roll No. |
|------|-------|
| `<Yashvi Prashant Jain>` | `<B24CM1086>` |
| `<Aanchal Bhaskar Shukla>` | `<B24CM1089>` |

---

## 1. Project Overview

Games that are too hard make players frustrated, and games that are too easy make them bored. Either way, they quit. **PlayFit** is an RL agent that performs *Dynamic Difficulty Adjustment (DDA)*. It watches how a player is doing (success, mistakes, response time, frustration, engagement) and adapts the game on the fly:

- **Enemy strength** (weaker / same / stronger)
- **Time limit** (tighter / same / looser)
- **Hints** (on / off)
- **Recovery opportunities** (on / off)

The reward encourages **sustained engagement and skill improvement** and penalizes **frustration, boredom, and simulated player dropout**.

The design follows *flow theory*: players engage and learn most when the challenge is slightly above their skill. "Always easy" and "always hard" are therefore both bad policies, and the agent must track a moving target, because skill grows during a session and differs between players.

**Algorithm:** Double + Dueling DQN with soft (Polyak) target updates, trained on a custom synthetic player simulator (`PlayFitEnv`). It is compared against a hand-tuned rule-based adaptive controller, a random policy, and three fixed-difficulty policies.

> This is a pure RL project. The only non-RL components are the baselines used for comparison in `evaluate.py`.

---

## 2. Repository Structure

```
.
├── environment.py     # PlayFitEnv: synthetic player simulator (the RL environment)
├── dqn_agent.py       # Double + Dueling DQN agent, replay buffer
├── train.py           # Training loop (+ validation / best-checkpoint selection)
├── evaluate.py        # Evaluation vs. baselines, CSVs and comparison plots
├── requirements.txt   # Python dependencies
├── README.md
└── artifacts/         # Generated outputs (trained agent, logs, plots)
    ├── playfit_dqn.pt            # Trained agent weights
    ├── training_log.csv          # Per-episode training stats
    ├── training_rewards.png      # Training curve
    ├── episode_results.csv       # Per-episode evaluation results (all policies)
    ├── evaluation_results.csv    # Summary: mean and 95% CI per policy
    └── *_comparison.png          # One bar chart per metric
```

---

## 3. Installation

**Requirements:** Python 3.9+ (tested on CPU only; no GPU needed).

```bash
# 1. Clone the repository
git clone https://github.com/yashvi-jain/PlayFit.git
cd PlayFit

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

| Package | Version |
|---------|---------|
| numpy | >= 1.24 |
| matplotlib | >= 3.7 |
| torch | >= 2.0 |

---

## 4. Execution Commands

### Train the agent

```bash
python train.py            # default seed 42  -> artifacts/playfit_dqn.pt
python train.py 7          # another seed     -> artifacts/playfit_dqn_seed7.pt
```

Training runs 3000 episodes, prints progress every 200 episodes, and saves the best greedy checkpoint (picked on held-out validation scenarios), the training log and the training-reward plot to `artifacts/`.

### Evaluate the agent against the baselines

```bash
python evaluate.py
```

This loads `artifacts/playfit_dqn.pt` (the seed-42 model) and runs 300 unseen player scenarios per policy. It prints a summary table, paired comparisons and DQN action diagnostics, and writes the CSVs and comparison plots to `artifacts/`.

> A pre-trained agent is included at `artifacts/playfit_dqn.pt`, so you can run `python evaluate.py` right away without retraining.

---

## 5. Reproducing Our Results

```bash
pip install -r requirements.txt
python train.py          # ~3000 episodes, saves artifacts/playfit_dqn.pt
python evaluate.py       # evaluates on 300 unseen scenarios
```

Reproducibility notes:

- **Training scenarios:** the environment for episode *k* is seeded with `k`.
- **Validation seeds:** 50000+, never used for training or evaluation.
- **Evaluation seeds:** 10000–10299, never used for training or validation.
- **Agent seed:** `python train.py <seed>` seeds Python, NumPy and PyTorch. The default is 42.
- PyTorch is limited to a single CPU thread, which keeps runs consistent.
- The CPU-only setup means small numerical differences across machines or PyTorch versions are possible. Expect the same trends rather than bit-identical numbers.

---

## 6. Notes

- The player is **simulated**, not human. 
- The environment encodes flow-theory assumptions, so results show that the agent adapts well within this model and should not be read as validated on real players.

---