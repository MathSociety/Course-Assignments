import numpy as np
import random
import gymnasium as gym
from mmwavebandits import mmWaveEnv
import tensorflow as tf
from tensorflow import keras

class EpsilonGreedyAgent:
    def __init__(self, env, epsilon=0.1):
        self.env = env
        self.epsilon = epsilon
        
        # Create a neural network model
        self.model = self.create_model()
        
    def create_model(self):
        """Create a simple neural network model to approximate action values"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(3,)),  # Adjust input shape based on observation size
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(2 * self.env.Nbeams)  # Output size is (Base Stations x Beams)
        ])
        model.compile(optimizer='adam', loss='mse')  # Mean Squared Error loss
        return model

    def select_action(self, obs):
        """Selects action based on epsilon-greedy policy"""
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: select random action
            bs_ix = np.random.choice([0, 1])
            beam_ix = np.random.choice(self.env.Nbeams)
        else:
            # Exploitation: use the model to select the action with the highest expected reward
            q_values = self.model.predict(np.array([obs]))[0]  # Predict Q-values for the given observation
            bs_ix, beam_ix = np.unravel_index(np.argmax(q_values), (2, self.env.Nbeams))
        
        return (bs_ix, beam_ix)
    
    def update_action_value(self, obs, action, reward, next_obs):
        """Update the Q-value using the Bellman equation"""
        bs_ix, beam_ix = action
        target = reward + 0.99 * np.max(self.model.predict(np.array([next_obs]))[0])  # Discount factor = 0.99
        target_f = self.model.predict(np.array([obs]))
        target_f[0][bs_ix * self.env.Nbeams + beam_ix] = target  # Update the target for the taken action
        self.model.fit(np.array([obs]), target_f, epochs=1, verbose=0)  # Train the model on the updated target
    
    def train(self, episodes):
        """Train the agent over a number of episodes"""
        rewards = []
        for ep in range(episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            done = False
            truncated = False
            while not (done or truncated):
                action = self.select_action(obs)
                next_obs, reward, done, truncated, _ = self.env.step(action)
                self.update_action_value(obs, action, reward, next_obs)
                obs = next_obs
                total_reward += reward
            rewards.append(total_reward)
        
        return rewards

# Initialize environment
env = mmWaveEnv()

# Initialize Epsilon-Greedy agent
agent = EpsilonGreedyAgent(env, epsilon=0.1)

# Train the agent
episodes = 1000  # Number of episodes to train on
rewards = agent.train(episodes)

# Print the learned action values (Base Station x Beams)
print("Learned Q-Values (Base Stations x Beams):")
print(agent.model.predict(np.array([obs])))

# Plotting the rewards if needed
import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Epsilon-Greedy Total Reward per Episode")
plt.show()

# Close the environment
env.close()
