import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

alpha = 0.9
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01
episodes = 20000
max_steps = 300

q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

for ep in range(episodes):
    state, _ = env.reset()
    for _ in range(max_steps):
        action = choose_action(state, epsilon)
        new_state, reward, terminated, truncated, _ = env.step(action)
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state]))
        state = new_state
        if terminated or truncated:
            break
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")

for _ in range(5):
    state, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)

env.close()
