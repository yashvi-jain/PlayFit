import random
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

class DuelingQNetwork(nn.Module):
    """Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)."""

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, action_dim))

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)

QNetwork = DuelingQNetwork

class ReplayBuffer:
    """Fixed-size ring buffer on preallocated numpy arrays (much faster than a deque of tuples)."""

    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, s, a, r, s2, d):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        ix = np.random.randint(0, self.size, size=n)
        return self.s[ix], self.a[ix], self.r[ix], self.s2[ix], self.d[ix]

    def __len__(self):
        return self.size

class DQNAgent:
    """Double + Dueling DQN with soft (Polyak) target updates."""

    def __init__(self, state_dim=9, action_dim=5, seed=42, lr=5e-4):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.action_dim = action_dim
        self.online = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

        self.memory = ReplayBuffer(50000, state_dim)
        self.gamma = 0.97
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9975
        self.learn_steps = 0
        self.tau = 0.005

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(self.online(s).argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        # `done` must mean a TRUE terminal (player dropped out). A session that merely hit
        # the time limit is a truncation: the future value is still there, so we bootstrap.
        self.memory.add(state, action, reward, next_state, float(done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, s2, d = self.memory.sample(self.batch_size)
        s = torch.from_numpy(s)
        a = torch.from_numpy(a).unsqueeze(1)
        r = torch.from_numpy(r).unsqueeze(1)
        s2 = torch.from_numpy(s2)
        d = torch.from_numpy(d).unsqueeze(1)

        current_q = self.online(s).gather(1, a)
        with torch.no_grad():
            best = self.online(s2).argmax(dim=1, keepdim=True)         # Double DQN: online picks
            next_q = self.target(s2).gather(1, best)                   # target evaluates
            target_q = r + self.gamma * next_q * (1.0 - d)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.learn_steps += 1

        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.mul_(1.0 - self.tau).add_(op, alpha=self.tau)
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({"model_state_dict": self.online.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(ckpt["model_state_dict"])
        self.target.load_state_dict(self.online.state_dict())
        self.epsilon = 0.0