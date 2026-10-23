import random
from collections import deque
import numpy as np
import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture — separate value and advantage streams.

    Splitting the Q-value into V(s) + A(s,a) lets the network learn which
    *states* are valuable independently of which *action* is best.  This is
    especially helpful when many actions have similar values (as in PlayFit,
    where Q-values differ by < 1 across actions).
    """

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Value stream — scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Advantage stream — per-action A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)                     # (B, 1)
        advantage = self.advantage_stream(features)             # (B, A)
        # Q = V + (A - mean(A))  [mean-centering for identifiability]
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


# Keep the old name available so existing code that imports QNetwork still
# works (e.g. unit tests, notebooks).  The actual architecture is upgraded.
QNetwork = DuelingQNetwork


class DQNAgent:
    def __init__(self, state_dim=8, action_dim=5, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.online = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        # --- Hyperparameter changes (documented) ---
        # Learning rate: 1e-3 → 5e-5
        #   Reason: Q-value differences across actions are < 0.5 on a ~45
        #   total-return scale.  A high LR causes the network to overshoot
        #   these tiny differences, collapsing to a single dominant action.
        #   5e-5 provides the precision needed to learn sub-unit advantages.
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=5e-5)

        self.memory = deque(maxlen=50000)
        self.gamma = 0.97
        # Batch size: 64 → 128
        #   Reason: larger batches produce lower-variance gradient estimates,
        #   essential when the signal (action advantage) is < 1% of the Q-scale.
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9975       # unchanged per requirement
        self.learn_steps = 0

        # Soft target update (Polyak averaging) with tau=0.005
        #   Reason: hard target-net copies every N steps cause oscillation
        #   in the target Q-values, which propagates through the Bellman
        #   backup and makes it impossible to stabilize the tiny action-
        #   advantage differences.  Continuous soft updates (θ_target ←
        #   τ·θ_online + (1-τ)·θ_target) smooth out the target landscape.
        self.tau = 0.005

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(state_tensor).argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        current_q = self.online(states).gather(1, actions)

        with torch.no_grad():
            # --- Double DQN ---
            # Standard DQN: next_q = target(s').max()
            #   Problem: the *same* network picks the best action AND evaluates
            #   it, systematically overestimating Q-values.  When true
            #   advantages are tiny (~0.3), overestimation washes them out.
            # Double DQN: online network SELECTS the best next-action,
            #   target network EVALUATES it.  This decouples selection from
            #   evaluation and preserves the small but real differences
            #   between actions.
            best_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, best_actions)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: 5.0 → 1.0
        #   Reason: tighter clipping prevents large gradient spikes from
        #   overwriting the small learned advantages.
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.learn_steps += 1

        # Soft target update every step
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save(self, path):
        torch.save({"model_state_dict": self.online.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(checkpoint["model_state_dict"])
        self.target.load_state_dict(self.online.state_dict())
        self.epsilon = 0.0
