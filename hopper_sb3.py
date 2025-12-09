import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import os




model_dir = "models"
logs_dir = "logs/hopper"
stats_path = os.path.join(model_dir, "vec_normalize.pkl")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)


device = "cpu"
TIMESTEPS = 30_000_000
ITERATIONS = 10


def train():

    env = make_vec_env("Hopper-v5", n_envs=12, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        'MlpPolicy',
        env=env,
        verbose=1,
        device=device,
        tensorboard_log=logs_dir,
        target_kl=0.03
    )

    local_timesteps = TIMESTEPS // ITERATIONS
    iters = 0

    for _ in range(ITERATIONS):
        iters += 1
        model.learn(total_timesteps=local_timesteps, reset_num_timesteps=False)

        model.save(f"{model_dir}/PPO_Hopper_{local_timesteps*iters}")
        env.save(stats_path)


def test():
    env = make_vec_env("Hopper-v5", n_envs=1, vec_env_cls=SubprocVecEnv)
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(f"{model_dir}/PPO_Hopper_{TIMESTEPS}")

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        env.render("human")



if __name__ == "__main__":
    train()