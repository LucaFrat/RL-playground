import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt


# --- HYPERPARAMETERS ---
LEARNING_RATE = 5e-4
TOTAL_TIMESTEPS = 20_000_000
NUM_ENVS = 32
STEPS_PER_ENV = 128
BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV
MINIBATCH_SIZE = 1024
PPO_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
ENTROPY_COEF = 0.0
MAX_GRAD_NORM = 0.5
ANNEAL_LR = True

USE_GPU = True

device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
print(f"Training Device: {device} | Parallel Envs: {NUM_ENVS}")


def make_env(env_id, seed, capture_video=False):
    def thunk():
        env = gym.make("Humanoid-v5", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        recording_config = {'is_eval': False}
        def video_trigger(episode_id):
            if episode_id % 100 == 0: return True
            if recording_config['is_eval']: return True
            return False

        # Only the first environment records video to avoid spam
        if capture_video and env_id == 0:
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder='./videos/humanoid_parallel',
                name_prefix='humanoid_vec',
                episode_trigger=video_trigger,
                disable_logger=True
            )

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512):
        super(MLP, self).__init__()
        self.actor_net = nn.Sequential(
            layer_init(nn.Linear(in_dim, hid_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hid_dim, hid_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hid_dim, out_dim), std=0.01)
        )
        self.critic_net = nn.Sequential(
            layer_init(nn.Linear(in_dim, hid_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hid_dim, hid_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hid_dim, 1), std=1.0)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, state):
        value_pred = self.critic_net(state)
        mu = self.actor_net(state)
        std = self.actor_log_std.exp().expand_as(mu)
        return mu, std, value_pred

def plot_train_rewards(train_rewards):
    plt.figure(figsize=(12,8))
    plt.plot(train_rewards, label="Average Episode Reward")
    plt.xlabel("Global Steps", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid()
    plt.legend()
    plt.show()

def evaluate_policy(env, agent, recording_config):
    # Ensure agent is in eval mode and on CPU for evaluation (simplest for single env)
    agent.eval()

    # Trigger recording
    recording_config['is_eval'] = True
    state, _ = env.reset()
    recording_config['is_eval'] = False # Reset immediately

    terminated, truncated = False, False
    true_reward = 0.0

    while not (terminated or truncated):
        # Hybrid: Eval on CPU is fine
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu, _, _ = agent(state)
            action = mu.squeeze(0).numpy() # Deterministic action for eval

        state, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            if "episode" in info:
                true_reward = info["episode"]["r"]
    return true_reward


# --- MAIN TRAINING LOOP ---
envs = gym.vector.AsyncVectorEnv(
    [make_env(i, i, capture_video=True) for i in range(NUM_ENVS)]
)

obs_space = envs.single_observation_space.shape[0]
action_space = envs.single_action_space.shape[0]


agent = MLP(obs_space, action_space).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

# Storage Buffers (Pre-allocated for speed) - (Steps, Envs, Dim)
obs_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS, obs_space)).to(device)
actions_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS, action_space)).to(device)
logprobs_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(device)
rewards_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(device)
dones_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(device)
values_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(device)

global_step = 0
start_time = time.time()
train_rewards = []
train_steps = []

# Initial Reset
next_obs, _ = envs.reset()
next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(NUM_ENVS).to(device)

num_updates = TOTAL_TIMESTEPS // BATCH_SIZE

print(f"Starting training for {num_updates} updates...")

for update in range(1, num_updates + 1):

    if ANNEAL_LR:
        fraction = 1.0 - (update - 1.0) / num_updates
        new_lr = fraction * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = new_lr

    # --- 1. DATA COLLECTION (Rollout) ---
    agent.eval()
    for step in range(STEPS_PER_ENV):
        global_step += NUM_ENVS

        # Save current observation and done
        obs_buffer[step] = next_obs
        dones_buffer[step] = next_done

        # Action Logic
        with torch.no_grad():
            mu, std, value = agent(next_obs)
            dist = Normal(mu, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(1)

        values_buffer[step] = value.flatten()
        actions_buffer[step] = action
        logprobs_buffer[step] = logprob

        action_cpu = action.cpu().numpy()
        real_next_obs, reward, terminations, truncations, infos = envs.step(action_cpu)
        # print(infos)

        next_done = np.logical_or(terminations, truncations)

        rewards_buffer[step] = torch.tensor(reward).to(device).view(-1)
        next_obs = torch.tensor(real_next_obs).to(device)
        next_done = torch.tensor(next_done, dtype=torch.float32).to(device)

        if "episode" in infos:
            true_rewards = infos['episode']['r']
            true_rewards = true_rewards[true_rewards > 0.1]

            if len(true_rewards) > 0:
                train_rewards.extend(true_rewards)
                train_steps.extend([global_step] * len(true_rewards))

                if len(train_rewards) % 500 == 0:
                     print(f"Step: {global_step} | Avg Reward: {np.mean(train_rewards[-50:]):.1f}")


    # --- 2. GAE CALCULATION ---
    with torch.no_grad():
        _, _, next_value = agent(next_obs)
        next_value = next_value.reshape(1, -1)

        advantages = torch.zeros_like(rewards_buffer).to(device)
        lastgaelam = 0

        for t in reversed(range(STEPS_PER_ENV)):
            if t == STEPS_PER_ENV - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value.flatten()
            else:
                nextnonterminal = 1.0 - dones_buffer[t + 1]
                nextvalues = values_buffer[t + 1]

            delta = rewards_buffer[t] + GAMMA * nextvalues * nextnonterminal - values_buffer[t]
            lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values_buffer

    # --- 3. FLATTEN BATCH ---
    # Flatten (Steps, Envs, ...) -> (Batch_Size, ...)
    b_obs = obs_buffer.reshape((-1, obs_space))
    b_logprobs = logprobs_buffer.reshape(-1)
    b_actions = actions_buffer.reshape((-1, action_space))
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values_buffer.reshape(-1)

    # --- 4. OPTIMIZATION (PPO Update) ---
    agent.train()
    b_inds = np.arange(BATCH_SIZE)
    clipfracs = []

    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(b_inds)

        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_inds = b_inds[start:end]

            _, _, newvalue = agent(b_obs[mb_inds])
            mu, std, _ = agent(b_obs[mb_inds])
            dist = Normal(mu, std)
            newlogprob = dist.log_prob(b_actions[mb_inds]).sum(1)
            entropy = dist.entropy().sum(1)

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()

            if approx_kl > 0.02:
                break

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss - ENTROPY_COEF * entropy_loss + 0.5 * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()



    # Log Speed
    if update % 10 == 0:
        sps = int(global_step / (time.time() - start_time))
        print(f"Update {update}/{num_updates} | SPS: {sps}")

envs.close()



env = gym.make("Humanoid-v5", render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ClipAction(env)
env = gym.wrappers.NormalizeObservation(env)
env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
env = gym.wrappers.NormalizeReward(env)
env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

def video_trigger(episode_id):
    return True

env = gym.wrappers.RecordVideo(
        env=env,
        video_folder='./videos/humanoid_parallel',
        name_prefix='humanoid_vec',
        episode_trigger= lambda episode_id: True,
        disable_logger=True
    )

agent.eval()
state, _ = env.reset()
terminated, truncated = False, False

while not (terminated or truncated):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _, _ = agent(state)
            action = mu.squeeze(0).cpu().numpy() # Deterministic action for eval

        state, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            if "episode" in info:
                true_reward = info["episode"]["r"]

print(f"Mean Eval Reward: {true_reward:.1f}")


env.close()


print()
print("---- ---- ---- ----")
print(f"TIME -> {time.time() - start_time} s.")
print("---- ---- ---- ----")
print()


# Plotting
plt.figure(figsize=(12,8))
plt.plot(train_steps, train_rewards, label="Training Reward")
plt.xlabel("Global Steps")
plt.ylabel("Reward")
plt.title("Vectorized PPO Training - Humanoid-v5")
plt.grid()
plt.show()