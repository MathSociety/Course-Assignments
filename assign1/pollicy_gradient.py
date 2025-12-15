import numpy as np
import matplotlib.pyplot as plt
from mmWave_bandits import mmWaveEnv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Define the policy model
def create_policy_model(input_dim, output_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))  # Fewer neurons
    model.add(Dense(12, activation='relu'))                       # Fewer neurons
    model.add(Dense(output_dim, activation='softmax'))  # Using softmax for action probabilities
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy')
    return model

# Function to sample action using the policy
def sample_action(policy_model, state):
    probabilities = policy_model.predict(state).flatten()
    return np.random.choice(len(probabilities), p=probabilities)

# Function to compute the average reward using a receding window
def compute_receding_window_average(rewards, window_size):
    avg_rewards = []
    for i in range(len(rewards)):
        if i < window_size:
            avg_rewards.append(np.mean(rewards[:i + 1]))
        else:
            avg_rewards.append(np.mean(rewards[i - window_size + 1:i + 1]))
    return avg_rewards

# Main training function with batch size concept
def train_policy_gradient(env, num_episodes, batch_size, learning_rate):
    input_dim = 12
    output_dim = 20
    policy_model = create_policy_model(input_dim, output_dim, learning_rate)
    
    rewards_list = []
    batch_states, batch_actions_one_hot, batch_rewards = [], [], []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        # Ensure obs is not empty before concatenation
        if obs is None or len(obs) == 0:
            print("Initial observation is empty. Exiting the episode.")
            break
        
        obs = np.concatenate(obs)
        num_zer = input_dim - obs.size
         
        if num_zer > 0:
            obs = np.pad(obs, (0, num_zer), mode='constant')
        obs = obs.reshape(1, -1)
        
        done = False
        episode_rewards = []
        actions = []
        states = []

        while not done:
            # Sample action from policy
            action = sample_action(policy_model, obs)

            # Step in the environment
            obs_next, reward, _, done, _ = env.step((action // 10, action % 10))

            # Check if obs_next is empty
            if obs_next is None or len(obs_next) == 0:
                print("Next observation is empty. Exiting the loop.")
                break

            # Store state, action, and reward
            states.append(obs)
            actions.append(action)
            episode_rewards.append(reward)
            rewards_list.append(reward)

            # Move to the next state
            obs = obs_next
            
            # Check if obs is empty before concatenation
            if obs is None or len(obs) == 0:
                print("Observation is empty after step. Exiting the loop.")
                break
            
            obs = np.concatenate(obs)
            num_zer = input_dim - obs.size
             
            if num_zer > 0:
                obs = np.pad(obs, (0, num_zer), mode='constant')
            obs = obs.reshape(1, -1)

        # Convert collected states and actions to numpy arrays
        states = np.array(states).reshape(-1, input_dim)
        actions_one_hot = np.zeros((len(actions), output_dim))
        actions_one_hot[np.arange(len(actions)), actions] = 1  # One-hot encoding of actions

        # Accumulate data for batch updates
        batch_states.append(states)
        batch_actions_one_hot.append(actions_one_hot)
        batch_rewards.append(episode_rewards)

        # Once batch is full, update the policy model
        if len(batch_states) >= batch_size:
            all_states = np.vstack(batch_states)
            all_actions_one_hot = np.vstack(batch_actions_one_hot)
            all_rewards = np.hstack(batch_rewards)

            # Use rewards as sample weights
            sample_weights = all_rewards

            # Train on the batch
            policy_model.train_on_batch(all_states, all_actions_one_hot, sample_weight=sample_weights)

            # Reset batch data
            batch_states, batch_actions_one_hot, batch_rewards = [], [], []

        # Optional: Print episode reward
        total_reward = sum(episode_rewards)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    return rewards_list, policy_model

# Main execution
if __name__ == "__main__":
    env = mmWaveEnv()
    num_episodes = 3  # Number of episodes for training
    batch_size = 1  # Set batch size
    learning_rate = 0.0001  # Set learning rate
    
    rewards, trained_policy = train_policy_gradient(env, num_episodes, batch_size=batch_size, learning_rate=learning_rate)

    # Compute and plot receding window average reward
    window_size = 10
    avg_rewards = compute_receding_window_average(rewards, window_size)

    plt.plot(avg_rewards, label='Receding Window Avg Reward')
    plt.title('Policy Gradient: Receding Window Average Reward')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.axhline(y=np.mean(rewards), color='r', linestyle='--', label='Overall Average Reward')
    plt.legend()
    plt.grid()
    plt.savefig('policy_gradient_rewards.png')
    plt.show()

    env.close()
