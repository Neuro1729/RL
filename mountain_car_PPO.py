import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("MountainCar-v0", render_mode="human")
n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

# Hyperparameters
lr = 1e-3
gamma = 0.99
epsilon = 0.2  # PPO clip parameter
episodes = 2000
max_steps = 200
epochs = 4  

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
    return action.item(), dist.log_prob(action), dist

# Training loop (PPO)
for ep in range(episodes):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    states = []
    actions = []

    for _ in range(max_steps):
        action, log_prob, dist = select_action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
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

    # Convert lists to tensors
    old_log_probs = torch.stack(log_probs)
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)

    # PPO update: multiple epochs over the same episode
    for _ in range(epochs):
        for idx in range(len(states_tensor)):
            state = states_tensor[idx]
            action = actions_tensor[idx]
            G = returns[idx]

            probs = policy_net(state)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)

            # Compute ratio (new / old)
            ratio = torch.exp(log_prob - old_log_probs[idx])

            # Clip ratio
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

            # PPO loss
            loss = -torch.min(ratio * G, clipped_ratio * G)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

env.close()
