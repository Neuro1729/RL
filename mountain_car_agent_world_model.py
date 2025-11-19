import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import matplotlib.pyplot as plt
import random

# -------------------------------
# World Model Definition
# -------------------------------
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(WorldModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        self.hidden_dim = hidden_dim

    def forward(self, states, actions, hidden=None):
        batch_size, seq_len, _ = states.shape
        encoded = []
        for t in range(seq_len):
            sa = torch.cat([states[:, t], actions[:, t]], dim=-1)
            encoded.append(self.encoder(sa))
        encoded = torch.stack(encoded, dim=1)  # (batch, seq, hidden)
        rnn_out, hidden = self.rnn(encoded, hidden)
        next_states_pred = self.decoder(rnn_out)
        return next_states_pred, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)

# -------------------------------
# Data Collector
# -------------------------------
class DataCollector:
    def __init__(self, state_dim, action_dim, max_size=10000):
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.next_states = deque(maxlen=max_size)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def add_transition(self, state, action, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)

    def get_dataset(self, sequence_length=10):
        states = np.array(self.states)
        actions = np.array(self.actions)
        next_states = np.array(self.next_states)
        states_seq, actions_seq, next_states_seq = [], [], []
        for i in range(len(states) - sequence_length):
            states_seq.append(states[i:i+sequence_length])
            actions_seq.append(actions[i:i+sequence_length])
            next_states_seq.append(next_states[i:i+sequence_length])
        return (
torch.FloatTensor(states_seq),
                torch.FloatTensor(actions_seq),
                torch.FloatTensor(next_states_seq))

# -------------------------------
# Collect training data
# -------------------------------
def collect_training_data(env, num_episodes=100, max_steps=200):
    collector = DataCollector(state_dim=2, action_dim=env.action_space.n)
    for ep in range(num_episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        for step in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)
            # One-hot encode action
            action_onehot = np.zeros(env.action_space.n)
            action_onehot[action] = 1
            collector.add_transition(state, action_onehot, next_state)
            state = next_state
            if done or truncated:
                break
    return collector

# -------------------------------
# Train World Model
# -------------------------------
def train_world_model(model, dataloader, num_epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for states, actions, next_states in dataloader:
            optimizer.zero_grad()
            next_states_pred, _ = model(states, actions)
            loss = criterion(next_states_pred, next_states)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    return train_losses

# -------------------------------
# Model-based Agent with Planning
# -------------------------------
def choose_action_with_world_model(model, state, action_space, plan_horizon=5, num_sequences=50, gamma=0.99):
    """
    Plan multiple random action sequences in the world model and pick best.
    """
    state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # (1,1,state_dim)
    best_return = -float('inf')
    best_action = None
    for _ in range(num_sequences):
        sim_state = state.clone()
        hidden = model.init_hidden(1)
        total_reward = 0
        actions_seq = []
        for t in range(plan_horizon):
            action = torch.randint(0, action_space.n, (1,))
            actions_seq.append(action)
            action_onehot = torch.zeros(1,1,action_space.n)
            action_onehot[0,0,action] = 1
            next_state_pred, hidden = model(sim_state, action_onehot, hidden)
            sim_state = next_state_pred
            # Reward function: MountainCar
            reward = sim_state[0,0,0].item()  # position
            total_reward += (gamma**t) * reward
        if total_reward > best_return:
            best_return = total_reward
            best_action = actions_seq[0].item()
    return best_action

# -------------------------------
# Run Model-based Agent
# -------------------------------
def run_model_based_agent(env, model, num_episodes=5, max_steps=200):
    for ep in range(num_episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        states = [state.copy()]
        for step in range(max_steps):
            action = choose_action_with_world_model(model, state, env.action_space)
            next_state, reward, done, truncated, _ = env.step(action)
            states.append(next_state.copy())
            state = next_state
            if done or truncated:
                break
        states = np.array(states)
        plt.plot(states[:,0], label=f'Episode {ep+1}')
    plt.title('MountainCar Position using Model-based Planning')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.legend()
    plt.show()

# -------------------------------
# Main
# -------------------------------
def main():
    env = gym.make('MountainCar-v0')
    print("Collecting training data...")
    collector = collect_training_data(env, num_episodes=100)
    states, actions, next_states = collector.get_dataset(sequence_length=10)
    dataset = TensorDataset(states, actions, next_states)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = WorldModel(state_dim=2, action_dim=env.action_space.n)
    print("Training World Model...")
    train_world_model(model, dataloader, num_epochs=50)

    print("Running model-based agent...")
    run_model_based_agent(env, model, num_episodes=3)

    env.close()

if __name__ == "__main__":
    main()
