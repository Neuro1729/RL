import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")

n_states = env.observation_space.n
n_actions = env.action_space.n

alpha = 0.1
gamma = 0.99
episodes = 5000
max_steps = 200

policy = np.ones((n_states, n_actions)) / n_actions

def choose_action(state):
    return np.random.choice(n_actions, p=policy[state])

for ep in range(episodes):
    state, _ = env.reset()
    episode_history = []

    for _ in range(max_steps):
        action = choose_action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        episode_history.append((state, action, reward))
        state = new_state
        if terminated or truncated:
            break

    G = 0
    for state, action, reward in reversed(episode_history):
        G = gamma * G + reward
        for a in range(n_actions):
            if a == action:
                policy[state, a] += alpha * G * (1 - policy[state, a])
            else:
                policy[state, a] -= alpha * G * policy[state, a]
        policy[state] = np.clip(policy[state], 1e-5, 1.0)
        policy[state] /= np.sum(policy[state])

for episode in range(5):
    state, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = choose_action(state)
        state, reward, terminated, truncated, _ = env.step(action)

env.close()
