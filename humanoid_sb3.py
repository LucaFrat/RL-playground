import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import os



model_dir = "models"
logs_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

device = "cpu"

def train(env):
    model = PPO('MlpPolicy', env=env, verbose=1, device=device, tensorboard_log=logs_dir)
    TIMESTEPS = 500_000
    iters = 0
    for _ in range(10):
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/PPO_{TIMESTEPS*iters}")



if __name__ == "__main__":
    env = make_vec_env("Humanoid-v5", n_envs=12, vec_env_cls=SubprocVecEnv)

    train(env=env)