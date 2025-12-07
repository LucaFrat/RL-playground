import torch
from torch import nn
import gymnasium as gym
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time





class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, dropout=0.2):
        super().__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.actor_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, state):
        value = self.critic_net(state)

        mu = self.actor_net(state)
        std = self.actor_log_std.exp().expand_as(mu)
        return mu, std, value




def create_agent(hidden_dim: int, dropout: float):
    INPUT_FEATURES = env_train.observation_space.shape[0]
    HIDDEN_DIMENSIONS = hidden_dim
    ACTOR_OUTPUT_FEATURES = env_train.action_space.shape[0]
    DROPOUT = dropout
    agent = ActorCritic(INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTOR_OUTPUT_FEATURES, DROPOUT)
    return agent


def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)

    returns = torch.tensor(returns)
    return returns


def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages


def calculate_surrogate_loss(
        actions_log_probability_old,
        actions_log_probability_new,
        epsilon,
        advantages):

    advantages = advantages.detach()
    policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()
    surrogate_loss1 = policy_ratio * advantages
    surrogate_loss2 = torch.clamp(policy_ratio, min=1.0-epsilon, max=1.0+epsilon) * advantages

    surrogate_loss = torch.min(surrogate_loss1, surrogate_loss2)
    return surrogate_loss


def calculate_losses(surrogate_loss, entropy, entropy_coeff, returns, value_pred):
    entropy_bonus = entropy * entropy_coeff
    policy_loss = -(surrogate_loss + entropy_bonus).sum()
    value_loss = f.smooth_l1_loss(returns, value_pred).sum()
    return policy_loss, value_loss


def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    terminated = False
    truncated = False
    episode_reward = 0
    return states, actions, actions_log_probability, values, rewards, terminated, truncated, episode_reward



def forward_pass(env, agent, discount_factor):
    states, actions, actions_log_prob, values, rewards, terminated, truncated, episode_reward = init_training()
    state, _ = env.reset()
    agent.train()

    while not (terminated or truncated):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        states.append(state)

        mu, std, value_pred = agent(state)
        dist = Normal(mu, std)
        action = dist.sample()

        log_prob_action = dist.log_prob(action).sum(dim=-1)
        action_numpy = action.squeeze(0).detach().cpu().numpy()
        state, reward, terminated, truncated, _ = env.step(action_numpy)

        actions.append(action)
        actions_log_prob.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward

    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    actions_log_prob = torch.cat(actions_log_prob).to(device)
    values = torch.cat(values).squeeze(-1).to(device)

    returns = calculate_returns(rewards, discount_factor).to(device)
    advantages = calculate_advantages(returns, values)

    return episode_reward, states, actions, actions_log_prob, advantages, returns



def update_policy(
        agent,
        states,
        actions,
        actions_log_prob_old,
        advantages,
        returns,
        optimizer,
        ppo_steps,
        epsilon,
        entropy_coeff):

    BATCH_SIZE = 128
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_prob_old = actions_log_prob_old.detach()
    actions = actions.detach()
    training_results_dataset = TensorDataset(
        states,
        actions,
        actions_log_prob_old,
        advantages,
        returns)
    batch_dataset = DataLoader(
        training_results_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False)

    for _ in range(ppo_steps):
        for batch_idx, (states, actions, actions_log_prob_old, advantages, returns) in enumerate(batch_dataset):
            mu, std, value_pred = agent(states)
            value_pred = value_pred.squeeze(-1)
            prob_distribution_new = Normal(mu, std)
            entropy = prob_distribution_new.entropy().sum(dim=-1)

            actions_log_prob_new = prob_distribution_new.log_prob(actions).sum(dim=-1)
            surrogate_loss = calculate_surrogate_loss(
                actions_log_prob_old,
                actions_log_prob_new,
                epsilon,
                advantages)
            policy_loss, value_loss = calculate_losses(
                surrogate_loss,
                entropy,
                entropy_coeff,
                returns,
                value_pred)
            optimizer.zero_grad()
            loss = policy_loss + value_loss
            loss.backward()

            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)

            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def evaluate(env, agent):
    agent.eval()
    rewards = []
    terminated = False
    truncated = False
    episode_reward = 0
    state, _ = env.reset()
    while not (terminated or truncated):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _, _ = agent(state)
            action = mu.squeeze(0).cpu().numpy()

        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
    return episode_reward



def plot_train_rewards(train_rewards, reward_threshold):
    plt.figure(figsize=(12,8))
    plt.plot(train_rewards, label="Training Reward")
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Training Reward", fontsize=20)
    plt.hlines(reward_threshold, 0, len(train_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def plot_test_rewards(test_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Testing Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Testing Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(test_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def plot_losses(policy_losses, value_losses):
    plt.figure(figsize=(12, 8))
    plt.plot(value_losses, label='Value Losses')
    plt.plot(policy_losses, label='Policy Losses')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()





def run_ppo():
    MAX_EPISODES = 20000
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 100
    PRINT_INTERVAL = 50
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 512
    DROPOUT = 0.2
    LEARNING_RATE = 0.0005
    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []
    agent = create_agent(HIDDEN_DIMENSIONS, DROPOUT).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for episode in range(1, MAX_EPISODES+1):

        train_reward, states, actions, actions_log_probability, advantages, returns = forward_pass(
                env_train,
                agent,
                DISCOUNT_FACTOR)

        policy_loss, value_loss = update_policy(
                agent,
                states,
                actions,
                actions_log_probability,
                advantages,
                returns,
                optimizer,
                PPO_STEPS,
                EPSILON,
                ENTROPY_COEFFICIENT)

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(train_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_INTERVAL == 0:
            print(f'Episode: {episode:3} | Mean Rewards: {mean_train_rewards:3.1f}')
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break
    plot_train_rewards(train_rewards, REWARD_THRESHOLD)
    plot_test_rewards(test_rewards, REWARD_THRESHOLD)
    plot_losses(policy_losses, value_losses)








if __name__ == '__main__':

    use_gpu = False
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Device: {device}")

    start_time = time.time()
    env_train = gym.make("Humanoid-v5", render_mode="rgb_array")

    env_train = gym.wrappers.ClipAction(env_train)
    env_train = gym.wrappers.NormalizeObservation(env_train)
    env_train = gym.wrappers.TransformObservation(env_train, lambda obs: np.clip(obs, -10, 10), env_train.observation_space)
    env_train = gym.wrappers.NormalizeReward(env_train)
    trigger = lambda episode_id: episode_id % 1999 == 0

    env_train = gym.wrappers.RecordVideo(
        env=env_train,
        video_folder='./video_logs',
        name_prefix='test_video',
        episode_trigger=trigger,
        disable_logger=True
    )

    run_ppo()

    env_train.close()

    print(f"TIME: {time.time() - start_time} sec.")







