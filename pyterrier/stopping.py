"""Trainable stopping rules with optional learning dependencies."""

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .transformer import Estimator


def _grlstop_dependencies():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as error:
        raise ImportError(
            "GRLStop requires optional dependencies; install pyterrier[stopping]"
        ) from error
    return gym, spaces, PPO, BaseCallback, Monitor, DummyVecEnv


@dataclass
class _GRLTrajectory:
    features: Any
    labels: np.ndarray
    bins: tuple
    states: dict = field(default_factory=dict)


class GRLStop(Estimator):
    """Generalised RLStop, trained with PPO on labelled ranked trajectories.

    Each state contains relevance rates in reviewed batches, logistic-regression
    predictions for unreviewed batches, the current batch, and target recall.
    Training uses the reward and 100-batch representation released with GRLStop.
    ``features`` is preferred for the embedded classifier; ``score`` is a
    one-dimensional fallback. Transforming a logged trajectory requires labels
    for its reviewed-prefix simulation.
    """

    def __init__(self, target_recall: float, train_target_recalls: Optional[Sequence[float]] = None,
                 n_windows: int = 100, total_timesteps: int = 20_000_000,
                 n_steps: int = 10, batch_size: Optional[int] = None,
                 learning_rate: float = .0003, ent_coef: float = .1,
                 gamma: float = .99, gae_lambda: float = .98, clip_range: float = .1,
                 n_epochs: int = 10, recall_power: float = 1, cost_power: float = 1,
                 random_state: int = 0, deterministic: bool = True):
        targets = tuple(train_target_recalls) if train_target_recalls is not None else (target_recall,)
        if (not 0 < target_recall <= 1 or not targets or any(not 0 < target <= 1 for target in targets)
                or n_windows < 2 or total_timesteps < 1 or n_steps < 1 or batch_size is not None and batch_size < 2
                or learning_rate <= 0 or ent_coef < 0 or not 0 < gamma <= 1 or not 0 < gae_lambda <= 1
                or clip_range <= 0 or n_epochs < 1 or recall_power <= 0 or cost_power <= 0):
            raise ValueError("invalid GRLStop parameter")
        self.target_recall = target_recall
        self.train_target_recalls = targets
        self.n_windows = n_windows
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.recall_power = recall_power
        self.cost_power = cost_power
        self.random_state = random_state
        self.deterministic = deterministic
        self.policy_ = None

    @staticmethod
    def _labelled(results, qrels=None):
        required = {'qid', 'docno', 'rank'}
        missing = required - set(results.columns)
        if missing:
            raise ValueError(f"ranked trajectories are missing columns: {sorted(missing)}")
        if 'label' in results.columns:
            labelled = results.copy()
        elif qrels is not None and {'qid', 'docno', 'label'} <= set(qrels.columns):
            labelled = results.merge(qrels[['qid', 'docno', 'label']], on=['qid', 'docno'], how='left')
        else:
            raise ValueError("labelled trajectories or qrels with qid, docno, and label are required")
        labelled['label'] = pd.to_numeric(labelled['label'], errors='raise').fillna(0)
        labelled['rank'] = pd.to_numeric(labelled['rank'], errors='raise')
        return labelled

    @staticmethod
    def _features(frame):
        if 'features' in frame.columns:
            values = frame['features'].tolist()
            if not values:
                raise ValueError("GRLStop cannot use an empty trajectory")
            from scipy import sparse
            matrix = sparse.vstack(values) if sparse.issparse(values[0]) else np.stack(values)
        elif 'score' in frame.columns:
            matrix = pd.to_numeric(frame['score'], errors='raise').to_numpy(dtype=float).reshape(-1, 1)
        else:
            raise ValueError("GRLStop needs a features column or ranking score for its classifier")
        return matrix

    def _trajectories(self, results, qrels=None):
        labelled = self._labelled(results, qrels)
        trajectories = []
        for qid, group in labelled.groupby('qid', sort=False):
            group = group.sort_values('rank', kind='stable').reset_index(drop=True)
            if len(group) < self.n_windows:
                raise ValueError(f"GRLStop needs at least {self.n_windows} ranked documents for qid {qid!r}")
            if group['docno'].duplicated().any():
                raise ValueError(f"GRLStop requires unique docno values for qid {qid!r}")
            features = self._features(group)
            bins = tuple(np.array_split(np.arange(len(group)), self.n_windows))
            trajectories.append((group, _GRLTrajectory(features, (group['label'].to_numpy() > 0).astype(int), bins)))
        return trajectories

    def _state(self, trajectory, position, target_recall):
        from sklearn.linear_model import LogisticRegression

        if position not in trajectory.states:
            state = np.empty(self.n_windows + 2, dtype=np.float32)
            observed_end = trajectory.bins[position][-1] + 1
            observed_labels = trajectory.labels[:observed_end]
            for index in range(position + 1):
                batch = trajectory.bins[index]
                state[index] = trajectory.labels[batch].mean()
            if position + 1 < self.n_windows:
                classes = np.unique(observed_labels)
                unseen_features = trajectory.features[observed_end:]
                if len(classes) == 1:
                    predicted = np.full(len(trajectory.labels) - observed_end, classes[0], dtype=int)
                else:
                    positives = int(observed_labels.sum())
                    negatives = len(observed_labels) - positives
                    class_weight = {1: negatives / positives, 0: 1} if positives >= negatives else {0: positives / negatives, 1: 1}
                    classifier = LogisticRegression(
                        solver='liblinear', random_state=self.random_state, C=1.0,
                        max_iter=10_000, class_weight=class_weight,
                    ).fit(trajectory.features[:observed_end], observed_labels)
                    predicted = classifier.predict(unseen_features)
                offset = 0
                for index in range(position + 1, self.n_windows):
                    size = len(trajectory.bins[index])
                    state[index] = predicted[offset:offset + size].mean()
                    offset += size
            trajectory.states[position] = state
        state = trajectory.states[position].copy()
        state[-2] = position
        state[-1] = target_recall
        return state

    def _target_position(self, trajectory, target_recall):
        total_relevant = trajectory.labels.sum()
        if not total_relevant:
            return self.n_windows - 1
        cumulative = 0
        for position, batch in enumerate(trajectory.bins):
            cumulative += trajectory.labels[batch].sum()
            if cumulative / total_relevant >= target_recall:
                return position
        return self.n_windows - 1

    def _reward(self, position, target_position):
        current = position + 1
        target = target_position + 1
        if current <= target:
            return (current ** self.recall_power - (current - 1) ** self.recall_power) / target ** self.recall_power
        if target == self.n_windows:
            return 0.0
        return ((self.n_windows - current) ** self.cost_power - (self.n_windows - current + 1) ** self.cost_power) / (self.n_windows - target) ** self.cost_power

    def _environment(self, trajectory, target_recall):
        gym, spaces, _, _, _, _ = _grlstop_dependencies()
        stopping_rule = self
        target_position = self._target_position(trajectory, target_recall)

        class Environment(gym.Env):
            def __init__(self):
                self.action_space = spaces.Discrete(2)
                self.observation_space = spaces.Box(-np.inf, np.inf, shape=(stopping_rule.n_windows + 2,), dtype=np.float32)
                self.position = 0
                self.first_continue = True

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.position = 0
                self.first_continue = True
                return stopping_rule._state(trajectory, self.position, target_recall), {}

            def step(self, action):
                action = int(action)
                if action not in (0, 1):
                    raise ValueError("GRLStop action must be continue (0) or stop (1)")
                terminated = action == 1
                if not terminated and self.position < stopping_rule.n_windows - 1:
                    # GRLStop's first continue action reviews the initial batch.
                    if self.first_continue:
                        self.first_continue = False
                    else:
                        self.position += 1
                truncated = not terminated and self.position == stopping_rule.n_windows - 1
                return (
                    stopping_rule._state(trajectory, self.position, target_recall),
                    stopping_rule._reward(self.position, target_position),
                    terminated,
                    truncated,
                    {},
                )

        return Environment

    def fit(self, topics_or_res_tr, qrels_tr=None, topics_or_res_va=None, qrels_va=None):
        _, _, PPO, BaseCallback, Monitor, DummyVecEnv = _grlstop_dependencies()
        trajectories = [trajectory for _, trajectory in self._trajectories(topics_or_res_tr, qrels_tr) if trajectory.labels.sum()]
        if not trajectories:
            raise ValueError("GRLStop needs at least one training trajectory with a relevant document")
        episodes = [(trajectory, target) for trajectory in trajectories for target in self.train_target_recalls]
        rollout_size = self.n_steps * len(episodes)
        if rollout_size < 2:
            raise ValueError("GRLStop needs at least two rollout samples; increase n_steps or training trajectories")
        batch_size = self.batch_size or max(2, rollout_size // 4)
        if self.batch_size is None:
            while batch_size > 2 and rollout_size % batch_size:
                batch_size -= 1
            if rollout_size % batch_size:
                batch_size = rollout_size
        environments = [self._environment(trajectory, target) for trajectory, target in episodes]

        class EarlyStopping(BaseCallback):
            def __init__(self):
                super().__init__()
                self.best = -np.inf
                self.stale = 0

            def _on_step(self):
                episode = self.locals['infos'][-1].get('episode')
                if episode is not None:
                    if episode['r'] > self.best + .001:
                        self.best, self.stale = episode['r'], 0
                    else:
                        self.stale += 1
                return self.stale <= 10 * self.model.n_steps * self.training_env.num_envs

        self.policy_ = PPO(
            'MlpPolicy', DummyVecEnv([lambda env=env: Monitor(env()) for env in environments]), n_steps=self.n_steps, batch_size=batch_size,
            learning_rate=self.learning_rate, ent_coef=self.ent_coef, gamma=self.gamma,
            gae_lambda=self.gae_lambda, clip_range=self.clip_range, n_epochs=self.n_epochs,
            policy_kwargs={'net_arch': [64, 64]}, seed=self.random_state, verbose=0, device='cpu',
        )
        self.policy_.learn(total_timesteps=self.total_timesteps, callback=EarlyStopping())
        return self

    def transform(self, inp):
        if self.policy_ is None:
            raise ValueError("GRLStop.fit() must be called before transform()")
        output = []
        for group, trajectory in self._trajectories(inp):
            stop = len(group)
            for position, batch in enumerate(trajectory.bins):
                action, _ = self.policy_.predict(self._state(trajectory, position, self.target_recall), deterministic=self.deterministic)
                if int(np.asarray(action).item()) == 1:
                    stop = batch[-1] + 1
                    break
            output.append(group.iloc[:stop])
        return pd.concat(output, ignore_index=True) if output else inp.copy()

    def save(self, path):
        if self.policy_ is None:
            raise ValueError("GRLStop.fit() must be called before save()")
        self.policy_.save(str(path))

    def load(self, path):
        _, _, PPO, _, _, _ = _grlstop_dependencies()
        self.policy_ = PPO.load(str(path))
        return self
