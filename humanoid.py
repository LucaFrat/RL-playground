import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time


def Train():

    # TRAIN
    env = gym.make("Humanoid-v5") #no rendering to speed up
    env = DummyVecEnv([lambda:env])

    print("Training START")
    model = PPO(policy="MlpPolicy", env=env, verbose=0,
                learning_rate=0.005, ent_coef=0.005, #exploration
                tensorboard_log="logs/") #>tensorboard --logdir=logs/
    model.learn(total_timesteps=300_000, #1h
                tb_log_name="model_humanoid", log_interval=10)
    print("Training DONE")


    model.save("model_humanoid")



def Test():
    env = gym.make("Humanoid-v5", render_mode="human")
    model = PPO.load(path="model_humanoid", env=env)
    obs, info = env.reset()

    reset = False #reset if the humanoid falls or the episode ends
    episode = 1
    total_reward, step = 0, 0

    for _ in range(1000):
        ## action
        step += 1
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ## reward
        total_reward += reward
        ## render
        env.render()
        time.sleep(1/240)
        if (step == 1) or (step % 100 == 0): #print first step and every 100 steps
            print(f"EPISODE {episode} - Step:{step}, Reward:{reward:.1f}, Total:{total_reward:.1f}")
        ## reset
        if reset:
            if terminated or truncated: #print the last step
                print(f"EPISODE {episode} - Step:{step}, Reward:{reward:.1f}, Total:{total_reward:.1f}")
                obs, info = env.reset()
                episode += 1
                total_reward, step = 0, 0
                print("------------------------------------------")

    env.close()


if __name__ == "__main__":
    start_time = time.time()
    Test()
    print(f"TOTAL TIME: {time.time() - start_time} sec.")
