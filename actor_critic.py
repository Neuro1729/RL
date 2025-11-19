import gym
import torch
import torch.nn as nn
import torch.optim as optim

# Environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Hyperparameters
lr = 1e-3
gamma = 0.99
episodes = 1000

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)         # Actor output
        value = self.critic(x)         # Critic output
        return logits, value

model = ActorCritic(state_dim, n_actions)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for ep in range(episodes):
    state = env.reset()
    state = torch.FloatTensor(state)
    done = False
    ep_reward = 0

    while not done:
        logits, value = model(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.FloatTensor(next_state)
        ep_reward += reward

        # Critic evaluation of next state
        _, next_value = model(next_state)
        advantage = reward + gamma * next_value * (1 - int(done)) - value

        # Losses
        actor_loss = -dist.log_prob(action) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    if (ep+1) % 50 == 0:
        print(f"Episode {ep+1}, Reward: {ep_reward}")
