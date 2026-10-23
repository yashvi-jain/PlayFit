import numpy as np


class PlayFitEnv:
    """Synthetic player model for RL-based dynamic difficulty adjustment (DDA).

    Design follows flow theory: a player stays engaged and learns fastest when
    challenge is *slightly above* skill. Both extremes are bad:
      * too hard  -> frustration  -> dropout
      * too easy  -> boredom      -> disengagement -> dropout, and almost no learning
    So "always easy" is no longer a free win, and the agent has to track a
    moving target (skill grows during the session, and differs per player).
    """

    N_OBS = 9
    SWEET_SPOT = -0.03     # challenge gap (difficulty - skill) where learning peaks
    SWEET_WIDTH = 0.20
    BORED_MARGIN = 0.15    # play is 'trivial' once difficulty sits this far below skill

    def __init__(self, seed=None, max_steps=100):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.action_deltas = np.array([-0.15, -0.07, 0.0, 0.07, 0.15], dtype=np.float32)
        self.reset()

    def reset(self):
        self.step_count = 0
        self.skill = float(self.rng.uniform(0.25, 0.75))
        self.initial_skill = self.skill
        self.difficulty = float(self.rng.uniform(0.35, 0.65))
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

    def _observation(self):
        trend = float(np.mean(self.success_history[-5:])) if self.success_history else 0.5
        # boredom is not directly visible in a real game; telemetry only gives a noisy signal
        boredom_obs = float(np.clip(self.boredom + self.rng.normal(0, 0.03), 0.0, 1.0))
        return np.array([
            self.skill, self.difficulty, self.last_success, self.last_mistakes,
            self.response_time, self.frustration, self.engagement, trend, boredom_obs
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() before step().")
        action = int(np.clip(action, 0, len(self.action_deltas) - 1))
        self.difficulty = float(np.clip(self.difficulty + self.action_deltas[action], 0.05, 0.95))
        challenge_gap = self.difficulty - self.skill
        support = 0.18 * (1.0 - self.difficulty)
        success_prob = float(np.clip(1.0 / (1.0 + np.exp(5.0 * (challenge_gap - support))), 0.05, 0.98))
        success = float(self.rng.random() < success_prob)
        mistakes = float(np.clip(1.0 - success + self.rng.normal(0.15, 0.12), 0.0, 1.0))
        response_time = float(np.clip(0.35 + 0.35 * max(challenge_gap, 0) + 0.25 * mistakes
                                      + self.rng.normal(0, 0.06), 0.05, 1.0))
        self.failure_streak = self.failure_streak + 1 if success < 0.5 else 0

        overload = max(0.0, challenge_gap)
        too_easy = max(0.0, -challenge_gap - self.BORED_MARGIN)      # how far below skill the level sits

        # too difficult
        frustration_change = (0.12 * mistakes + 0.15 * overload + 0.05 * self.failure_streak
                              - 0.12 * success - 0.08 * (1.0 - self.difficulty))
        self.frustration = float(np.clip(0.78 * self.frustration + frustration_change, 0.0, 1.0))

        # trivially easy
        self.boredom = float(np.clip(0.85 * self.boredom + 0.9 * too_easy - 0.05 * max(0.0, challenge_gap),
                                     0.0, 1.0))

        zpd = float(np.exp(-((challenge_gap - self.SWEET_SPOT) / self.SWEET_WIDTH) ** 2))
        skill_gain = 0.016 * self.learn_rate * zpd * (0.4 + 0.6 * success) * (0.5 + self.engagement)
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
            - (5.0 if dropped_out else 0.0)
        )
        info = {
            "success": success, "frustration": self.frustration, "boredom": self.boredom,
            "engagement": self.engagement, "skill": self.skill,
            "difficulty": self.difficulty, "dropout": dropped_out,
            "truncated": bool(self.done and not dropped_out),
        }
        return self._observation(), float(reward), self.done, info