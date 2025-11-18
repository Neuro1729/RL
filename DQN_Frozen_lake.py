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

q1 = np.zeros((env.observation_space.n, env.action_space.n))
q2 = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q1[state] + q2[state])

for ep in range(episodes):
    state, _ = env.reset()
    for _ in range(max_steps):
        action = choose_action(state, epsilon)
        new_state, reward, terminated, truncated, _ = env.step(action)

        if random.random() < 0.5:
            a = np.argmax(q1[new_state])
            q1[state, action] = (1 - alpha) * q1[state, action] + alpha * (reward + gamma * q2[new_state, a])
        else:
            a = np.argmax(q2[new_state])
            q2[state, action] = (1 - alpha) * q2[state, action] + alpha * (reward + gamma * q1[new_state, a])

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
        action = np.argmax(q1[state] + q2[state])
        state, reward, terminated, truncated, _ = env.step(action)

env.close()
