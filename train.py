import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from dropout_wrapper import SensorDropoutWrapper

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 100_000
DROPOUT_RATE    = 0.3


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_interval=2000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.timestep_log = []
        self.reward_log = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_interval == 0 and self.episode_rewards:
            mean_r = float(np.mean(self.episode_rewards[-20:]))
            self.timestep_log.append(self.num_timesteps)
            self.reward_log.append(mean_r)
            if self.verbose:
                print(f"  step {self.num_timesteps:>7} | mean_reward={mean_r:.1f}")
        return True


def make_env(dropout_rate=0.0):
    env = gym.make("CartPole-v1")
    if dropout_rate > 0:
        env = SensorDropoutWrapper(env, dropout_rate=dropout_rate)
    env = Monitor(env)
    return env


def train_agent(name, dropout_rate=0.0):
    print(f"\n{'='*50}")
    print(f" Training: {name}  (dropout={dropout_rate})")
    print(f"{'='*50}")

    env = make_env(dropout_rate)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
    )

    callback = RewardLoggerCallback(log_interval=2000, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    save_path = os.path.join(MODELS_DIR, name)
    model.save(save_path)
    print(f" Saved to {save_path}.zip")
    env.close()
    return callback


def save_training_logs(cb_clean, cb_robust):
    path = os.path.join(MODELS_DIR, "training_logs.npz")
    np.savez(
        path,
        clean_steps=np.array(cb_clean.timestep_log),
        clean_rewards=np.array(cb_clean.reward_log),
        robust_steps=np.array(cb_robust.timestep_log),
        robust_rewards=np.array(cb_robust.reward_log),
    )
    print(f" Training logs saved to {path}")


if __name__ == "__main__":
    cb_clean  = train_agent("ppo_clean",  dropout_rate=0.0)
    cb_robust = train_agent("ppo_robust", dropout_rate=DROPOUT_RATE)
    save_training_logs(cb_clean, cb_robust)
    print("\n Done! Run: python evaluate.py")