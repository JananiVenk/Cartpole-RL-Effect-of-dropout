import gymnasium as gym
import numpy as np


class SensorDropoutWrapper(gym.ObservationWrapper):
    """
    Randomly zeros out each observation dimension independently
    with probability `dropout_rate` at every step.

    CartPole obs: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    """

    def __init__(self, env, dropout_rate: float = 0.3):
        super().__init__(env)
        assert 0.0 <= dropout_rate < 1.0
        self.dropout_rate = dropout_rate

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.dropout_rate == 0.0:
            return obs
        mask = (np.random.random(obs.shape) > self.dropout_rate).astype(obs.dtype)
        return obs * mask


class SingleDimDropout(gym.ObservationWrapper):
    """
    Always zeros out one specific observation dimension.
    Used to measure per-sensor importance.

    dim: 0=cart_pos, 1=cart_vel, 2=pole_angle, 3=pole_ang_vel
    """

    DIM_NAMES = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]

    def __init__(self, env, dim: int):
        super().__init__(env)
        assert 0 <= dim < 4
        self.dim = dim

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        obs[self.dim] = 0.0
        return obs