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

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64):
        super(MLP, self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, out_dim)
        )
        self.critic_net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, state):
        value_pred = self.critic_net(state)
        mu = self.actor_net(state)
        std = self.actor_log_std.exp().expand_as(mu)
        return mu, std, value_pred

def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)
    return torch.tensor(returns, dtype=torch.float32)

def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def calculate_surrogate_loss(actions_log_prob_old, actions_log_prob_new, advantages, epsilon):
    policy_ratio = (actions_log_prob_new - actions_log_prob_old).exp()
    surrogate_loss_1 = policy_ratio * advantages
    surrogate_loss_2 = torch.clamp(policy_ratio, 1-epsilon, 1+epsilon) * advantages
    return torch.min(surrogate_loss_1, surrogate_loss_2)

def calculate_losses(surrogate_loss, entropy, returns, value_pred, entropy_coeff):
    entropy_bonus = entropy * entropy_coeff
    policy_loss = -(surrogate_loss + entropy_bonus).mean()
    value_loss = F.mse_loss(value_pred, returns)
    return policy_loss, value_loss

def plot_train_rewards(train_rewards):
    plt.figure(figsize=(12,8))
    plt.plot(train_rewards, label="True Training Reward")
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Reward", fontsize=20)
    plt.grid()
    plt.show()

def evaluate_policy(env, agent):
    agent.eval()

    recording_config['is_eval'] = True
    state, _ = env.reset()
    recording_config['is_eval'] = False

    terminated, truncated = False, False

    while not (terminated or truncated):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _, _ = agent(state)
            action = mu.squeeze(0).cpu().numpy()
        state, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if "episode" in info:
                true_reward = info["episode"]["r"]
    return true_reward



LEARNING_RATE = 3e-4
NUM_EPISODES = 5_000
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
PPO_STEPS = 10
EPSILON = 0.2
ENTROPY = 0.0
MIN_BUFFER_SIZE = 2048


USE_GPU = False
device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
print(f"Device: {device}")

env = gym.make("Hopper-v5", render_mode="rgb_array")

env = gym.wrappers.RecordEpisodeStatistics(env)

env = gym.wrappers.ClipAction(env)
env = gym.wrappers.NormalizeObservation(env)
env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
env = gym.wrappers.NormalizeReward(env)
env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

recording_config = {
    'is_eval': False
}
def video_trigger(episode_id):
    # Record every 1000th episode (Training)
    if episode_id % 1000 == 0:
        return True
    # OR Record if we are manually forcing it (Evaluation)
    if recording_config['is_eval']:
        return True
    return False

env = gym.wrappers.RecordVideo(
    env=env,
    video_folder='./video_logs',
    name_prefix='test_video',
    episode_trigger=video_trigger,
    disable_logger=True
)
obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
agent = MLP(obs_space, action_space).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

# Buffer to store trajectories across episodes
buffer = {
    'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []
}
train_rewards = []
start_time = time.time()

print("Starting training...")

for episode in range(NUM_EPISODES+1):
    state, _ = env.reset()
    terminated, truncated = False, False

    # Temporary buffer for the current episode
    ep_states, ep_actions, ep_log_probs, ep_rewards, ep_values = [], [], [], [], []

    while not (terminated or truncated):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            mu, std, value_pred = agent(state_tensor)
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob_action = dist.log_prob(action).sum(dim=-1)

        action_numpy = action.squeeze(0).cpu().numpy()
        next_state, reward, terminated, truncated, info = env.step(action_numpy)

        # Store step data
        ep_states.append(state_tensor.cpu())
        ep_actions.append(action.cpu())
        ep_log_probs.append(log_prob_action.cpu())
        ep_rewards.append(reward)
        ep_values.append(value_pred.cpu())

        state = next_state

        # Logging: If episode ended, capture the TRUE reward from RecordEpisodeStatistics
        if terminated or truncated:
            if "episode" in info:
                true_reward = info["episode"]["r"]
                train_rewards.append(true_reward)
                if len(train_rewards) % 20 == 0:
                     print(f"Episode {episode} | Avg Reward: {np.mean(train_rewards[-20:]):.1f}")

    # Process episode data
    # Calculate returns/advantages for this episode immediately
    ep_returns = calculate_returns(ep_rewards, DISCOUNT_FACTOR)
    ep_values = torch.cat(ep_values).squeeze(-1)
    ep_advantages = calculate_advantages(ep_returns, ep_values)

    # Add to global buffer
    buffer['states'].append(torch.cat(ep_states).to(device))
    buffer['actions'].append(torch.cat(ep_actions).to(device))
    buffer['log_probs'].append(torch.cat(ep_log_probs).to(device))
    buffer['rewards'].append(ep_returns)     # Storing returns directly
    buffer['values'].append(ep_values)
    buffer['dones'].append(ep_advantages)    # Storing advantages directly

    # Check total collected steps
    total_steps = sum(len(t) for t in buffer['states'])

    # UPDATE NETWORK Only if we have enough data (Stable PPO)
    if total_steps >= MIN_BUFFER_SIZE:

        # Flatten the buffer
        b_states = torch.cat(buffer['states'])
        b_actions = torch.cat(buffer['actions'])
        b_log_probs = torch.cat(buffer['log_probs'])
        b_returns = torch.cat(buffer['rewards'])
        b_advantages = torch.cat(buffer['dones'])

        # Create Dataset
        dataset = TensorDataset(b_states, b_actions, b_log_probs, b_advantages, b_returns)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(PPO_STEPS):
            for b_s, b_a, b_lp_old, b_adv, b_ret in loader:

                b_s = b_s.to(device)
                b_a = b_a.to(device)
                b_lp_old = b_lp_old.to(device)
                b_adv = b_adv.to(device)
                b_ret = b_ret.to(device)

                mu, std, value_pred = agent(b_s)
                value_pred = value_pred.squeeze(-1)

                dist = Normal(mu, std)
                new_log_probs = dist.log_prob(b_a).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                surrogate_loss = calculate_surrogate_loss(b_lp_old, new_log_probs, b_adv, EPSILON)
                policy_loss, value_loss = calculate_losses(surrogate_loss, entropy, b_ret, value_pred, ENTROPY)

                optimizer.zero_grad()
                loss = policy_loss + value_loss
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Clear buffer after update
        for k in buffer: buffer[k] = []

eval_rewards = []
for _ in range(5):
    episode_reward = evaluate_policy(env, agent)
    eval_rewards.append(episode_reward)

print(f"Mean Eval Reward: {np.average(eval_rewards):.1f}")

env.close()
print()
print("---- ---- ---- ----")
print(f"TIME -> {time.time() - start_time} s.")
print("---- ---- ---- ----")
print()
plot_train_rewards(train_rewards)
