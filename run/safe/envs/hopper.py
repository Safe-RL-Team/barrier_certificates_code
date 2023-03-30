from gym.envs import register
from gym.envs.mujoco.hopper_v3 import HopperEnv
from .safe_env_spec import SafeEnv, interval_barrier
from gym.utils.ezpickle import EzPickle
import numpy as np


class SafeHopperEnv(HopperEnv, SafeEnv):
    episode_unsafe = False

    def __init__(self, threshold=1, task='zoom', random_reset=False, violation_penalty=20):
        self.threshold = threshold
        self.task = task
        self.random_reset = random_reset
        self.violation_penalty = violation_penalty
        super().__init__()
        EzPickle.__init__(self, threshold=threshold, task=task, random_reset=random_reset)  # deepcopy calls `get_state`

    def reset_model(self):
        if self.random_reset:
            qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
            qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
            self.set_state(qpos, qvel)
        else:
            qvel = self.init_qvel
            qpos = self.init_qpos
            self.set_state(qpos, qvel)
        self.episode_unsafe = False
        return self._get_obs()

    def _get_obs(self):
        return super()._get_obs().astype(np.float32)

    def step(self, a):
        a = np.clip(a, -1, 1)
        next_state, _, done, info = super().step(a)

        if self.task == 'zoom':
            reward = (next_state[6] ** 2) * np.sign(next_state[6])
        else:
            assert 0

        if abs(next_state[..., 0]) < 0.7:
            # breakpoint()
            self.episode_unsafe = True
            reward -= self.violation_penalty
        info['episode.unsafe'] = self.episode_unsafe
        return next_state, reward, self.episode_unsafe, info

    def is_state_safe(self, states):
        return self.barrier_fn(states) <= 1.0

    def barrier_fn(self, states):
        return interval_barrier(states[..., 1], -self.threshold, self.threshold).maximum(interval_barrier(states[..., 0], -0.9, 0.9))

    def reward_fn(self, states, actions, next_states):
        return -(next_states[..., 0] ** 2 + next_states[..., 1] ** 2 + next_states[..., 2] ** 2) - actions[..., 0] ** 2 * 0.01

register('SafeHopper-v0', entry_point=SafeHopperEnv, max_episode_steps=1000)
