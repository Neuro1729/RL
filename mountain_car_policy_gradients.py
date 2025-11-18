import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Environment
env = gym.make("MountainCar-v0", render_mode="human")
n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

# Hyperparameters
lr = 1e-2
gamma = 0.99
episodes = 2000
max_steps = 200

# Policy network
class PolicyNN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

policy_net = PolicyNN(state_dim, n_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

# Function to select action
def select_action(state):
    state = torch.FloatTensor(state)
    probs = policy_net(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

# Training loop
for ep in range(episodes):
    state, _ = env.reset()
    log_probs = []
    rewards = []

    for _ in range(max_steps):
        action, log_prob = select_action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = new_state
        if terminated or truncated:
            break

    # Compute discounted returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient update
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env.close()
