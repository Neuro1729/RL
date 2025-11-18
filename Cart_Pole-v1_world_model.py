import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CartPole-v1", render_mode="human")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
latent_dim = 8

# --- VAE ---
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2*latent_dim)  # mu and logvar
        )
    def forward(self, x):
        x = self.fc(x)
        mu, logvar = x[:, :latent_dim], x[:, latent_dim:]
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
    def forward(self, z):
        return self.fc(z)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z

vae = VAE().to(device)
vae_opt = optim.Adam(vae.parameters(), lr=1e-3)
recon_loss_fn = nn.MSELoss(reduction='sum')

# --- RNN world model ---
class LatentRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim + n_actions, 32, batch_first=True)
        self.fc = nn.Linear(32, latent_dim)
    def forward(self, z, actions, h=None):
        # z: (batch, seq, latent_dim)
        # actions: (batch, seq, n_actions one-hot)
        inp = torch.cat([z, actions], dim=-1)
        out, h = self.rnn(inp, h)
        z_next = self.fc(out)
        return z_next, h

rnn = LatentRNN().to(device)
rnn_opt = optim.Adam(rnn.parameters(), lr=1e-3)
rnn_loss_fn = nn.MSELoss()

# --- Training loop ---
episodes = 1000
max_steps = 200
vae_epochs = 5

# 1) Train VAE first
obs_list = []
for ep in range(episodes):
    obs, _ = env.reset()
    for _ in range(max_steps):
        obs_list.append(obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

obs_tensor = torch.FloatTensor(obs_list).to(device)
for epoch in range(vae_epochs):
    x_hat, mu, logvar, _ = vae(obs_tensor)
    recon_loss = recon_loss_fn(x_hat, obs_tensor)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_loss
    vae_opt.zero_grad()
    loss.backward()
    vae_opt.step()
    print(f"VAE Epoch {epoch+1}, Loss: {loss.item():.2f}")

# 2) Train RNN on latent sequences
seq_len = 10
z_seq = []
action_seq = []

obs_array = obs_tensor.numpy()
for i in range(len(obs_array)-seq_len-1):
    seq_obs = torch.FloatTensor(obs_array[i:i+seq_len]).to(device)
    x_hat, mu, logvar, z = vae(seq_obs)
    z_seq.append(z[:-1])
    actions = np.random.randint(0, n_actions, seq_len)  # random actions for now
    actions_onehot = np.eye(n_actions)[actions[:-1]]
    action_seq.append(torch.FloatTensor(actions_onehot).to(device))

z_seq = torch.stack(z_seq)
action_seq = torch.stack(action_seq)
z_target = z_seq[:,1:,:]  # next latent

for epoch in range(5):
    z_pred, _ = rnn(z_seq[:,:-1,:], action_seq)
    loss = rnn_loss_fn(z_pred, z_target)
    rnn_opt.zero_grad()
    loss.backward()
    rnn_opt.step()
    print(f"RNN Epoch {epoch+1}, Loss: {loss.item():.2f}")

# --- Controller (simple random for demonstration) ---
state, _ = env.reset()
h = None
for t in range(max_steps):
    obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    _, _, _, z = vae(obs_tensor)
    action = env.action_space.sample()
    # Predict next latent (just demonstration)
    z_next, h = rnn(z.unsqueeze(0), torch.nn.functional.one_hot(torch.tensor([action]), n_actions).float().unsqueeze(0), h)
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()
