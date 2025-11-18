import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

# CEM hyperparameters
population_size = 50
elite_frac = 0.2
episodes_per_eval = 1
max_steps = 100
iterations = 100

# Initialize policy probabilities randomly for each state
policy_mean = np.ones((n_states, n_actions)) / n_actions
policy_std = 0.3  # initial noise

for iteration in range(iterations):
    # Sample a population of policies
    population = []
    rewards_population = []

    for _ in range(population_size):
        policy_sample = policy_mean + policy_std * np.random.randn(n_states, n_actions)
        policy_sample = np.clip(policy_sample, 1e-5, None)
        policy_sample /= policy_sample.sum(axis=1, keepdims=True)
        population.append(policy_sample)

        # Evaluate this policy
        total_reward = 0
        for _ in range(episodes_per_eval):
            state, _ = env.reset()
            done = False
            truncated = False
            for _ in range(max_steps):
                action = np.random.choice(n_actions, p=policy_sample[state])
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
        rewards_population.append(total_reward / episodes_per_eval)

    # Select elite policies
    elite_num = int(population_size * elite_frac)
    elite_idx = np.argsort(rewards_population)[-elite_num:]
    elite_policies = [population[i] for i in elite_idx]

    # Update mean and std
    policy_mean = np.mean(elite_policies, axis=0)
    policy_std = np.std(elite_policies, axis=0)

    print(f"Iteration {iteration+1}, elite avg reward: {np.mean([rewards_population[i] for i in elite_idx])}")

# Test the final policy
state, _ = env.reset()
done = False
truncated = False
total_reward = 0
while not (done or truncated):
    action = np.random.choice(n_actions, p=policy_mean[state])
    state, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    env.render()

print("Total reward:", total_reward)
env.close()
