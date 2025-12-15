import numpy as np
from mmWave_bandits import mmWaveEnv

class EpsilonGreedyAgent:
    def __init__(self, n_actions, n_contexts, epsilon=0.1):
        self.n_actions = n_actions  # Total number of actions (e.g., the size of the action space)
        self.n_contexts = n_contexts  # Total number of contexts
        self.epsilon = epsilon  # Probability of choosing a random action
        self.q_values = np.zeros((self.n_actions, self.n_contexts))  # Initialize Q-values for each action and context
        self.action_counts = np.zeros((self.n_actions, self.n_contexts))  # Count of times each action has been taken for each context

    def select_action(self, observation):
        if np.random.rand() < self.epsilon:
            # Explore: select a random action
            return (np.random.randint(2), np.random.randint(0, self.n_actions))  # Choose random beam action
        else:
            # Exploit: select the action with the highest Q-value for the current context
            context_index = self.get_context_index(observation)  # Define how to extract context from observation
            action_index = np.argmax(self.q_values[:, context_index])  # Best action based on current Q-values
            # Return the best action in the same format as your action space (e.g., (beam_action, beam_index))
            return (0, action_index)  # Assuming beam action is 0

    def update(self, action, reward, observation):
        context_index = self.get_context_index(observation)  # Get current context index
        action_index = action[1]  # Assuming action format is (beam_action, beam_index)
        
        # Update the count for the action taken
        self.action_counts[action_index, context_index] += 1
        
        # Update the Q-value using the incremental formula
        self.q_values[action_index, context_index] += (reward - self.q_values[action_index, context_index]) / self.action_counts[action_index, context_index]

    def get_context_index(self, observation):
        # This method should define how to extract the context from the observation
        # Here, you can use a simple mapping; for example:
        # If your observation has Ncars, you could create a mapping based on car positions, etc.
        # For simplicity, we'll assume it returns a fixed index or computes it based on the first feature.
        return min(int(observation[0][0] // 10), self.n_contexts - 1)  # Just a simple heuristic

# Main Loop to interact with the environment
def main():
    env = mmWaveEnv()
    n_actions =10  # Assuming it has a second space for beams
    n_contexts =4  # Based on the size of the observation
    agent = EpsilonGreedyAgent(n_actions, n_contexts, epsilon=0.1)

    for episode in range(1000):  # Run for a number of episodes
        observation, _ = env.reset()  # Reset the environment
        total_reward = 0

        for t in range(env.Horizon):
            action = agent.select_action(observation)  # Select an action based on the epsilon-greedy strategy
            next_observation, reward, terminated, truncated, _ = env.step(action)  # Step through the environment
            agent.update(action, reward, observation)  # Update the agent's Q-values based on the received reward
            total_reward += reward
            observation = next_observation  # Update the current observation

            if terminated or truncated:
                break

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

if __name__ == "__main__":
    main()
