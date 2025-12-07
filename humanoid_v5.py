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
            layer_init(nn.Linear(hid_dim, out_dim), std=0.01) # Reduced gain for initial stability
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

def calculate_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        last_advantage = delta + gamma * lam * mask * last_advantage
        advantages.insert(0, last_advantage)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    # Returns are calculated as Advantage + Value
    returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
    return advantages, returns

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

# --- HYPERPARAMETERS ---
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 10_000_000  # INCREASED: Humanoid needs ~2M+ steps
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 1024            # Increased for GPU efficiency
PPO_STEPS = 10
EPSILON = 0.2
ENTROPY = 0.0
MIN_BUFFER_SIZE = 4096

# Define device but don't move agent yet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Device: {device}")

env = gym.make("Humanoid-v5", render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ClipAction(env)
env = gym.wrappers.NormalizeObservation(env)
env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
env = gym.wrappers.NormalizeReward(env)
env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

recording_config = {'is_eval': False}
def video_trigger(episode_id):
    if episode_id % 2000 == 0: return True
    if recording_config['is_eval']: return True
    return False

env = gym.wrappers.RecordVideo(
    env=env,
    video_folder='./videos/humanoid',
    name_prefix='humanoid_agent',
    episode_trigger=video_trigger,
    disable_logger=True
)

obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

# HYBRID: Initialize Agent on CPU (Faster Collection)
agent = MLP(obs_space, action_space).to("cpu")
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

buffer = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'advantages': []}
train_rewards = []
start_time = time.time()

print("Starting training...")

global_counter = 0
episode_num = 0

while global_counter < TOTAL_TIMESTEPS:
    state, _ = env.reset()
    terminated, truncated = False, False
    episode_num += 1

    ep_states, ep_actions, ep_log_probs, ep_rewards, ep_values = [], [], [], [], []

    # --- COLLECTION PHASE (CPU) ---
    while not (terminated or truncated):
        state_tensor = torch.FloatTensor(state).unsqueeze(0) # Keep on CPU

        with torch.no_grad():
            mu, std, value_pred = agent(state_tensor)
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob_action = dist.log_prob(action).sum(dim=-1)

        action_numpy = action.squeeze(0).numpy()
        next_state, reward, terminated, truncated, info = env.step(action_numpy)
        global_counter += 1

        # Store purely on CPU
        ep_states.append(state_tensor)
        ep_actions.append(action)
        ep_log_probs.append(log_prob_action)
        ep_rewards.append(reward)
        ep_values.append(value_pred)

        state = next_state

        if terminated or truncated:
            if "episode" in info:
                true_reward = info["episode"]["r"]
                train_rewards.append(true_reward)
                if len(train_rewards) % 2_000 == 0:
                     print(f"Step: {global_counter} | Episode {episode_num} | Avg Reward: {np.mean(train_rewards[-20:]):.1f}")

        # Safety break
        if global_counter >= TOTAL_TIMESTEPS:
            break

    # Bootstrap value for GAE
    with torch.no_grad():
        _, _, next_value = agent(torch.FloatTensor(state).unsqueeze(0))
        next_value = next_value.item()
    ep_values.append(next_value)

    # GAE Calculation
    ep_dones = [0] * (len(ep_rewards) - 1) + [1]
    ep_advantages, ep_returns = calculate_gae(ep_rewards, ep_values, ep_dones)
    ep_advantages = (ep_advantages - ep_advantages.mean()) / (ep_advantages.std() + 1e-8)

    # Store in global buffer (CPU)
    buffer['states'].append(torch.cat(ep_states))
    buffer['actions'].append(torch.cat(ep_actions))
    buffer['log_probs'].append(torch.cat(ep_log_probs))
    buffer['rewards'].append(ep_returns)
    buffer['advantages'].append(ep_advantages)

    # --- UPDATE PHASE (GPU) ---
    total_steps = sum(len(t) for t in buffer['states'])

    if total_steps >= MIN_BUFFER_SIZE:
        # 1. Move Agent to GPU
        agent.to(device)

        # 2. Prepare Batch on GPU
        b_states = torch.cat(buffer['states']).to(device)
        b_actions = torch.cat(buffer['actions']).to(device)
        b_log_probs = torch.cat(buffer['log_probs']).to(device)
        b_returns = torch.cat(buffer['rewards']).to(device)
        b_advantages = torch.cat(buffer['advantages']).to(device)

        dataset = TensorDataset(b_states, b_actions, b_log_probs, b_advantages, b_returns)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(PPO_STEPS):
            for batch in loader:
                b_s, b_a, b_lp_old, b_adv, b_ret = batch

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

        # 3. Move Agent back to CPU for collection
        agent.to("cpu")

        # Clear buffer
        for k in buffer: buffer[k] = []

# Final Evaluation
eval_rewards = []
for _ in range(2):
    episode_reward = evaluate_policy(env, agent, recording_config)
    eval_rewards.append(episode_reward)

print(f"Mean Eval Reward: {np.average(eval_rewards):.1f}")
env.close()
print()
print("---- ---- ---- ----")
print(f"TIME -> {time.time() - start_time} s.")
print("---- ---- ---- ----")
print()
plot_train_rewards(train_rewards)