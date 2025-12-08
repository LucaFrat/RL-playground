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
TIMESTEPS = 25_000_000
ITERATIONS = 10

def train():
    env = make_vec_env("Humanoid-v5", n_envs=12, vec_env_cls=SubprocVecEnv)
    model = PPO('MlpPolicy', env=env, verbose=1, device=device, tensorboard_log=logs_dir)
    local_timesteps = TIMESTEPS//ITERATIONS
    iters = 0
    for _ in range(ITERATIONS):
        iters += 1
        model.learn(total_timesteps=local_timesteps, reset_num_timesteps=False)
        model.save(f"{model_dir}/PPO_{local_timesteps*iters}")

def test():
    env = make_vec_env("Humanoid-v5", n_envs=4, vec_env_cls=SubprocVecEnv)

    model = PPO.load(f"{model_dir}/PPO_{TIMESTEPS}")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated = env.step(action)
        env.render("human")


if __name__ == "__main__":
    train()