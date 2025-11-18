import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

alpha = 0.1      # learning rate
gamma = 0.99     # discount factor
epsilon = 0.2    # PPO clip parameter
episodes = 5000
max_steps = 200

# Policy table: probability distribution over actions for each state
policy = np.ones((n_states, n_actions)) / n_actions

# Value table: estimate of expected return for each state
value = np.zeros(n_states)

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

    # Compute discounted returns
    G = 0
    returns = []
    for state, action, reward in reversed(episode_history):
        G = reward + gamma * G
        returns.insert(0, G)
    
    # Convert to numpy array
    returns = np.array(returns)

    for (state, action, _), G in zip(episode_history, returns):
        # Advantage = G - value estimate
        advantage = G - value[state]
        value[state] += alpha * advantage  # update value table

        old_prob = policy[state, action]
        # Compute new probability using clipped PPO update
        new_prob = old_prob + alpha * advantage
        new_prob = np.clip(new_prob, old_prob*(1-epsilon), old_prob*(1+epsilon))
        # Normalize policy
        policy[state, action] = new_prob
        policy[state] = np.clip(policy[state], 1e-5, 1.0)
        policy[state] /= np.sum(policy[state])

# Test the trained policy
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")

for _ in range(5):
    state, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = np.argmax(policy[state])
        state, reward, terminated, truncated, _ = env.step(action)

env.close()
