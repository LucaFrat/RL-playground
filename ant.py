import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym



env = gym.make("Hopper-v5", render_mode="rgb_array")



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512):
        super(MLP, self).__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
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
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

        self.actor_log_std = nn.Parameter(torch.ones(1, out_dim) * -0.5)

    def forward(self, state):
        value_pred = self.critic_net(state)

        mu = self.actor_net(state)
        std = self.actor_log_std.exp().expand_as(mu)
        return mu, std, value_pred



def init_training_episode():
    action_log_prob = 0
    truncated = False
    terminated = False
    states = []
    actions = []
    values = []
    rewards = []
    episode_reward = 0
    return rewards, states, actions, terminated, truncated, values, action_log_prob, episode_reward


def calculate_returns(rewards, discount_factor):
    returns = []
    cumuluative_reward = 0
    for reward in reversed(rewards):
        cumuluative_reward = reward + cumuluative_reward * discount_factor
        returns.insert(0, cumuluative_reward)
    return torch.tensor(returns)


def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages




# hyperparameters:
learning_rate = 0.005
num_episodes = 100
discount_factor = 0.99

# initialize network
obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
agent = MLP(obs_space, action_space)

optimizer = optim.Adam(agent.parameters(), lr=learning_rate)


# FORWARD PASS ------------------------------------------------------------
for episode in range(num_episodes):

    agent.train()
    state, info = env.reset()
    reward = 0
    rewards, stated, actions, terminated, truncated, values, action_log_prob, episode_reward = init_training_episode()

    while not (truncated or terminated):
        state = torch.FloatTensor(state).unsqueeze(0)

        mu, std, value_pred = agent(state)
        dist = Normal(mu, std)
        action = dist.sample()

        log_prob_action = dist.log_prob(action).sum(dim=-1)
        action_numpy = action.squeeze(0).detach().cpu().numpy()

        state, reward, terminated, truncated, _ = env.step(action_numpy)

        stated.append(state)
        actions.append(action)
        action_log_prob.append(log_prob_action)
        values.append(value_pred)
        rewards.append(rewards)
        episode_reward += reward

    states = torch.cat(states)
    actions = torch.cat(actions)
    values = torch.cat(values)
    action_log_prob = torch.cat(action_log_prob)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)


# BACKWARD PASS ------------------------------------------------------------
# calculate loss (value and agent)
# compute gradient and backpropagate
# update the netword for PPO_steps on the same episode



