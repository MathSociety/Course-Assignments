import numpy as np
import pandas as pd
import gymnasium as gym
from tensorflow.keras import layers, models, optimizers
import os
import json

# Function to load previous rewards if they exist
def load_rewards(filename="rewards.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            rewards = json.load(file)
        return rewards
    else:
        return []

# Function to save rewards to a file after each episode
def save_rewards(rewards, filename="rewards.json"):
    with open(filename, 'w') as file:
        json.dump(rewards, file)

def find_min_score(path, percentile=50):
    """
    Calculate the minimum score based on a given percentile of total rewards in the dataset.

    Args:
        path (str): Path to the CSV file containing the dataset.
        percentile (float): Percentile of the total reward to use as the min_score.

    Returns:
        float: Calculated min_score based on the dataset.
    """
    dataset = pd.read_csv(path)
    dataset_group = dataset.groupby('Play #')
    total_rewards = [np.sum(df.iloc[:, 3].astype(np.float32)) for _, df in dataset_group]
    min_score = np.percentile(total_rewards, percentile)
    return min_score


def save_training_progress(episode, epsilon):
    save_path = "training_progress.json"
    with open(save_path, 'w') as f:
        json.dump({'episode': episode, 'epsilon': epsilon}, f)


def load_training_progress(save_path="training_progress.json"):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        with open(save_path, 'r') as f:
            progress = json.load(f)
        return progress['episode'], progress['epsilon']
    else:
        # If the file doesn't exist or is empty, initialize it with default values
        initial_data = {'episode': 0, 'epsilon': 1.0}
        with open(save_path, 'w') as f:
            json.dump(initial_data, f)
        return 0, 1.0


def create_model(state_dim, action_dim):
    """
    Neural network for Double DQN, using state and ordinal action as input to predict Q-value.
    Architecture: 2 hidden layers with 64 neurons each.
    """
    state_input = layers.Input(shape=(state_dim,), name='state_input')
    action_input = layers.Input(shape=(1,), name='action_input')  # Action as ordinal input (single integer)

    # Process state
    state_layer = layers.Dense(64, activation='relu')(state_input)
    state_layer = layers.Dense(64, activation='relu')(state_layer)

    # Process action
    action_layer = layers.Embedding(input_dim=action_dim, output_dim=16)(action_input)  # Embedding for actions
    action_layer = layers.Flatten()(action_layer)
    action_layer = layers.Dense(64, activation='relu')(action_layer)
    action_layer = layers.Dense(64, activation='relu')(action_layer)

    # Combine state and action
    combined = layers.Concatenate()([state_layer, action_layer])
    combined_layer = layers.Dense(64, activation='relu')(combined)
    combined_layer = layers.Dense(64, activation='relu')(combined_layer)

    # Output: Predicted Q-value (reward)
    output = layers.Dense(1, activation='linear')(combined_layer)

    model = models.Model(inputs=[state_input, action_input], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def load_offline_data(path, min_score):
    """
    Load and filter offline data based on the minimum score threshold.
    """
    state_data, action_data, reward_data, next_state_data, terminated_data = [], [], [], [], []
    dataset = pd.read_csv(path)
    dataset_group = dataset.groupby('Play #')

    for play_no, df in dataset_group:
        state = np.array([np.fromstring(row[1:-1], dtype=np.float32, sep=' ') for row in df.iloc[:, 1]])
        action = df.iloc[:, 2].astype(int).to_numpy()
        reward = df.iloc[:, 3].astype(np.float32).to_numpy()
        next_state = np.array([np.fromstring(row[1:-1], dtype=np.float32, sep=' ') for row in df.iloc[:, 4]])
        terminated = df.iloc[:, 5].astype(int).to_numpy()

        total_reward = np.sum(reward)
        if total_reward >= min_score:
            state_data.append(state)
            action_data.append(action)
            reward_data.append(reward)
            next_state_data.append(next_state)
            terminated_data.append(terminated)

    state_data = np.concatenate(state_data)
    action_data = np.concatenate(action_data)
    reward_data = np.concatenate(reward_data)
    next_state_data = np.concatenate(next_state_data)
    terminated_data = np.concatenate(terminated_data)

    return state_data, action_data, reward_data, next_state_data, terminated_data
def DDQN_training(env, offline_data, use_offline_data):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize networks
    predict_dqn = create_model(state_dim, action_dim)
    target_dqn = create_model(state_dim, action_dim)
    target_dqn.set_weights(predict_dqn.get_weights())

    replay_buffer = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "terminateds": [],
    }

    if use_offline_data:
        replay_buffer["states"] = list(offline_data[0])
        replay_buffer["actions"] = list(offline_data[1])
        replay_buffer["rewards"] = list(offline_data[2])
        replay_buffer["next_states"] = list(offline_data[3])
        replay_buffer["terminateds"] = list(offline_data[4])

    # Hyperparameters
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    batch_size = 16  # Reduced batch size for faster training
    update_target_frequency = 5  # Update target network more frequently
    save_model_frequency = 1  # Save the model every episode
    episodes = 100  # Increased episodes
    E=50
    
    total_rewards = load_rewards()  # Load previous rewards if available

    # Load training progress
    start_episode, epsilon = load_training_progress()

    if start_episode != 0:
        start_episode += 1


    for episode in range(start_episode, episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:  # Removed the max steps constraint
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = [predict_dqn.predict([np.expand_dims(state, axis=0), np.array([[a]])], verbose=0) for a in range(action_dim)]
                action = np.argmax(q_values)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if episode >= E:
                replay_buffer["states"].append(state)
                replay_buffer["actions"].append(action)
                replay_buffer["rewards"].append(reward)
                replay_buffer["next_states"].append(next_state)
                replay_buffer["terminateds"].append(1 if terminated else 0)

            state = next_state

            # Train network using random batch
            if len(replay_buffer["states"]) >= batch_size:
                indices = np.random.choice(len(replay_buffer["states"]), batch_size, replace=False)
                states = np.array([replay_buffer["states"][i] for i in indices])
                actions = np.array([replay_buffer["actions"][i] for i in indices])
                rewards = np.array([replay_buffer["rewards"][i] for i in indices])
                next_states = np.array([replay_buffer["next_states"][i] for i in indices])
                terminateds = np.array([replay_buffer["terminateds"][i] for i in indices])

                # Compute target Q-values
                target_q_values = target_dqn.predict([next_states, actions], verbose=0)
                targets = rewards + gamma * (1 - terminateds) * np.max(target_q_values, axis=1)

                # Update predict_dqn
                predict_dqn.train_on_batch([states, actions], targets)

        total_rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network
        if episode % update_target_frequency == 0:
            target_dqn.set_weights(predict_dqn.get_weights())
        
        save_rewards(total_rewards)

        # Save training progress and model periodically
        save_training_progress(episode, epsilon)
        if episode % save_model_frequency == 0:
            #predict_dqn.save(f"lunar_lander_model.keras")
            predict_dqn.save("lunar_lander_modelofflineTRUE.keras")

            print(f"Model saved at episode {episode}")

        print(f"Episode {episode}/{episodes}, Total Reward: {total_rewards[-1]}")

    save_rewards(total_rewards)

    # Save the final model
    #predict_dqn.save("lunar_lander_model.keras")
    predict_dqn.save("lunar_lander_modelofflineTRUE.keras")

    print("Final model saved")

    return predict_dqn, total_rewards



# Plot reward per episode and moving average
def plot_reward(total_reward_per_episode, window_length):
    import matplotlib.pyplot as plt
    moving_avg = pd.Series(total_reward_per_episode).rolling(window=window_length).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(total_reward_per_episode, label='Total Reward')
    plt.plot(moving_avg, label=f'{window_length}-Episode Moving Average', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.show()


# Main Execution
env = gym.make("LunarLander-v2", render_mode=None)

# Find minimum score dynamically
path = 'lunar_dataset.csv'
min_score = find_min_score(path, percentile=50)

# Load offline data
offline_data = load_offline_data(path, min_score)

# Train the DDQN model
use_offline_data = True
final_model, total_reward_per_episode = DDQN_training(env, offline_data, use_offline_data)

# Save the final model
#final_model.save("lunar_lander_model.keras")   #for offline data use =false
final_model.save("lunar_lander_modelofflineTRUE.keras")  
plot_reward(total_reward_per_episode, window_length=50)

env.close()
