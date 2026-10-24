import numpy as np


class PlayFitEnv:
    """Synthetic player model for RL-based dynamic difficulty adjustment (DDA).

    The agent controls four things each step: enemy strength, time limit, hints and recovery
    opportunities (e.g. checkpoints / health drops after a failure). Action index layout:
        a = ((enemy_idx * 3 + time_idx) * 2 + hint) * 2 + recovery

    Design follows flow theory: a player stays engaged and learns fastest when
    challenge is *slightly above* skill. Both extremes are bad:
      * too hard  -> frustration  -> dropout
      * too easy  -> boredom      -> disengagement -> dropout, and almost no learning
    So "always easy" is no longer a free win, and the agent has to track a
    moving target (skill grows during the session, and differs per player).
    """

    N_OBS = 12
    # --- controllable game knobs (the agent's action space is their Cartesian product) ---
    ENEMY_DELTAS = np.array([-0.10, 0.0, 0.10], dtype=np.float32)   # enemy strength change per step
    TIME_DELTAS = np.array([-0.10, 0.0, 0.10], dtype=np.float32)    # time-limit change (1.0 = generous, 0.2 = tight)
    N_ACTIONS = len(ENEMY_DELTAS) * len(TIME_DELTAS) * 2 * 2        # x hint on/off x recovery on/off = 36
    NO_OP = 16                                                       # enemy 0, time 0, hint off, recovery off
    TIME_PRESSURE_W = 0.20   # a tight time limit raises effective challenge
    HINT_RELIEF = 0.10       # an active hint lowers effective challenge ...
    HINT_LEARN_PENALTY = 0.30  # ... but a player who leans on hints learns less
    SWEET_SPOT = -0.03     # challenge gap (difficulty - skill) where learning peaks
    SWEET_WIDTH = 0.20
    BORED_MARGIN = 0.15    # play is 'trivial' once difficulty sits this far below skill

    def __init__(self, seed=None, max_steps=100):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.reset()

    @classmethod
    def encode_action(cls, enemy_idx, time_idx, hint, recovery):
        return ((int(enemy_idx) * 3 + int(time_idx)) * 2 + int(hint)) * 2 + int(recovery)

    @classmethod
    def decode_action(cls, action):
        recovery = action % 2
        hint = (action // 2) % 2
        time_idx = (action // 4) % 3
        enemy_idx = action // 12
        return enemy_idx, time_idx, hint, recovery

    def reset(self):
        self.step_count = 0
        self.skill = float(self.rng.uniform(0.25, 0.75))
        self.initial_skill = self.skill
        self.enemy_strength = float(self.rng.uniform(0.35, 0.65))
        self.time_limit = 0.7          # normalised time allowed per challenge
        self.hint = 0                  # hint currently shown (0/1)
        self.recovery = 0              # recovery opportunity currently offered (0/1)
        self.difficulty = self._effective_difficulty()
        self.frustration = float(self.rng.uniform(0.05, 0.25))
        self.boredom = float(self.rng.uniform(0.0, 0.10))
        self.engagement = 0.75
        # hidden per-player trait: how fast this player learns (agent must infer it)
        self.learn_rate = float(self.rng.uniform(0.7, 1.3))
        self.last_success = 0.5
        self.last_mistakes = 0.5
        self.response_time = 0.5
        self.success_history = []
        self.failure_streak = 0
        self.done = False
        return self._observation()

    def _effective_difficulty(self):
        """What the player actually feels: enemy strength + time pressure - hint relief."""
        return self.enemy_strength + self.TIME_PRESSURE_W * (1.0 - self.time_limit) - self.HINT_RELIEF * self.hint

    def _observation(self):
        trend = float(np.mean(self.success_history[-5:])) if self.success_history else 0.5
        # boredom is not directly visible in a real game; telemetry only gives a noisy signal
        boredom_obs = float(np.clip(self.boredom + self.rng.normal(0, 0.03), 0.0, 1.0))
        return np.array([
            self.skill, self.enemy_strength, self.time_limit, float(self.hint), float(self.recovery),
            self.last_success, self.last_mistakes, self.response_time,
            self.frustration, self.engagement, trend, boredom_obs
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() before step().")
        action = int(np.clip(action, 0, self.N_ACTIONS - 1))
        e_idx, t_idx, hint, recovery = self.decode_action(action)
        self.enemy_strength = round(float(np.clip(self.enemy_strength + self.ENEMY_DELTAS[e_idx], 0.05, 0.95)), 4)
        self.time_limit = round(float(np.clip(self.time_limit + self.TIME_DELTAS[t_idx], 0.2, 1.0)), 4)
        self.hint, self.recovery = hint, recovery
        time_pressure = 1.0 - self.time_limit
        self.difficulty = self._effective_difficulty()
        challenge_gap = self.difficulty - self.skill
        # recovery opportunities (checkpoints, health drops) make a failed attempt less costly
        support = 0.18 * (1.0 - self.difficulty) + 0.06 * self.recovery
        success_prob = float(np.clip(1.0 / (1.0 + np.exp(5.0 * (challenge_gap - support))), 0.05, 0.98))
        success = float(self.rng.random() < success_prob)
        mistakes = float(np.clip(1.0 - success + 0.15 * time_pressure + self.rng.normal(0.15, 0.12), 0.0, 1.0))
        response_time = float(np.clip(0.35 + 0.35 * max(challenge_gap, 0) + 0.25 * mistakes + 0.15 * time_pressure
                                      + self.rng.normal(0, 0.06), 0.05, 1.0))
        self.failure_streak = self.failure_streak + 1 if success < 0.5 else 0

        overload = max(0.0, challenge_gap)
        too_easy = max(0.0, -challenge_gap - self.BORED_MARGIN)      # how far below skill the level sits

        # too difficult
        frustration_change = (0.12 * mistakes + 0.15 * overload + 0.10 * time_pressure
                              + 0.05 * self.failure_streak * (1.0 - 0.5 * self.recovery)
                              - 0.12 * success - 0.08 * (1.0 - self.difficulty)
                              - 0.10 * self.recovery * (1.0 - success) - 0.03 * self.hint)
        self.frustration = float(np.clip(0.78 * self.frustration + frustration_change, 0.0, 1.0))

        # trivially easy
        self.boredom = float(np.clip(0.85 * self.boredom + 0.9 * too_easy - 0.05 * max(0.0, challenge_gap)
                                     + 0.02 * self.hint + 0.02 * self.recovery,
                                     0.0, 1.0))

        zpd = float(np.exp(-((challenge_gap - self.SWEET_SPOT) / self.SWEET_WIDTH) ** 2))
        skill_gain = 0.016 * self.learn_rate * zpd * (0.4 + 0.6 * success) * (0.5 + self.engagement) \
                     * (1.0 - self.HINT_LEARN_PENALTY * self.hint)
        self.skill = float(np.clip(self.skill + skill_gain, 0.05, 0.95))

        self.engagement = float(np.clip(
            0.88 * self.engagement + 0.18 * success - 0.22 * self.frustration
            - 0.30 * self.boredom + self.rng.normal(0, 0.025), 0.0, 1.0
        ))

        self.last_success = success
        self.last_mistakes = mistakes
        self.response_time = response_time
        self.success_history.append(success)
        self.step_count += 1

        # players quit from frustration as well as from boredom
        dropout_prob = float(np.clip(
            0.005 + 0.20 * self.frustration + 0.05 * (self.failure_streak >= 3) + 0.10 * self.boredom,
            0.005, 0.40))
        dropped_out = bool(self.rng.random() < dropout_prob)
        self.done = dropped_out or self.step_count >= self.max_steps

        reward = (
            0.5 * success + 0.4 * self.engagement + 20.0 * skill_gain
            - 0.75 * self.frustration - 0.60 * self.boredom - 0.25 * mistakes
            - 0.08 * self.hint - 0.05 * self.recovery            # small cost: don't spam assists
            - (5.0 if dropped_out else 0.0)
        )
        info = {
            "success": success, "frustration": self.frustration, "boredom": self.boredom,
            "engagement": self.engagement, "skill": self.skill,
            "difficulty": self.difficulty, "enemy_strength": self.enemy_strength,
            "time_limit": self.time_limit, "hint": self.hint, "recovery": self.recovery,
            "dropout": dropped_out,
            "truncated": bool(self.done and not dropped_out),
        }
        return self._observation(), float(reward), self.done, info