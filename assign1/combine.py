import numpy as np
import matplotlib.pyplot as plt
import math
from mmWave_bandits import mmWaveEnv
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Function for Linear Greedy
def lin_greedy(env, d, n):
    Naction = 20
    Ncontext = 12
    theta = np.zeros((Naction, Ncontext + 1))
    A = np.zeros((Naction, Ncontext + 1, Ncontext + 1))
    b = np.zeros((Naction, Ncontext + 1))

    Nepisodes = 10
    reward_arr = []

    for _ in range(Nepisodes):
        t = 0
        obs, _ = env.reset()
        z = np.concatenate(obs).reshape(-1, 1)
        z = np.pad(z, ((0, 12 - z.shape[0]), (0, 0)), 'constant', constant_values=0)
        z = np.concatenate((z, np.array([[1]])), axis=0)

        truncated = False
        while t < 720 and not truncated:
            epsilon = max(d / (t + 1) ** n, 0.01)
            v = np.random.uniform(0, 1)

            if v <= epsilon:
                action = np.random.randint(Naction)
            else:
                action = np.argmax(np.matmul(theta, z))

            obs_next, reward, _, truncated, _ = env.step((action // 10, action % 10))
            reward_arr.append(reward)

            if not truncated:
                z_temp = z.reshape((-1, 1))
                A[action] += np.matmul(z_temp, np.transpose(z_temp))
                b[action] += reward * z_temp.reshape(-1)
                theta[action] = np.matmul(np.linalg.inv(A[action] + 0.01 * np.eye(Ncontext + 1)), b[action])

                obs = obs_next
                z = np.concatenate(obs).reshape(-1, 1)
                z = np.pad(z, ((0, 12 - z.shape[0]), (0, 0)), 'constant', constant_values=0)
                z = np.concatenate((z, np.array([[1]])), axis=0)

            t += 1

    return np.array(reward_arr)

# Function for Linear UCB
def lin_ucb(env, d, n):
    Naction = 20
    Nfeatures = 12
    delta = 0.02
    theta = np.zeros((Naction, Nfeatures + 1))
    A = np.zeros((Naction, Nfeatures + 1, Nfeatures + 1))
    I = np.eye(Nfeatures + 1)
    for i in range(Naction):
        A[i] += delta * I
    b = np.zeros((Naction, Nfeatures + 1))

    Nepisodes = 10
    reward_arr = []

    for _ in range(Nepisodes):
        t = 0
        obs, _ = env.reset()
        z = np.concatenate(obs).reshape(-1, 1)
        z = np.pad(z, ((0, 12 - z.shape[0]), (0, 0)), 'constant', constant_values=0)
        z = np.concatenate((z, np.array([[1]])), axis=0)

        truncated = False
        while t < 720 and not truncated:
            epsilon = max(d / (t + 1) ** n, 0.01)
            values = []
            for i in range(Naction):
                A_inv = np.linalg.inv(A[i])
                firstpart = z.T @ theta[i].reshape(-1, 1)
                second = z.T @ A_inv @ z
                sqrt_term = np.sqrt(second[0, 0])
                value = firstpart[0, 0] + (epsilon * sqrt_term)
                values.append(value)

            action = np.argmax(values)
            obs_next, reward, _, truncated, _ = env.step((action // 10, action % 10))
            reward_arr.append(reward)

            if not truncated:
                z_temp = z.reshape((-1, 1))
                A[action] += np.matmul(z_temp, np.transpose(z_temp))
                b[action] += reward * z_temp.reshape(-1)
                theta[action] = np.matmul(np.linalg.inv(A[action] + 0.01 * np.eye(Nfeatures + 1)), b[action])

                obs = obs_next
                z = np.concatenate(obs).reshape(-1, 1)
                z = np.pad(z, ((0, 12 - z.shape[0]), (0, 0)), 'constant', constant_values=0)
                z = np.concatenate((z, np.array([[1]])), axis=0)
            t += 1

    return np.array(reward_arr)

# Function for Policy Gradient
def create_policy_model(input_dim, output_dim, learning_rate=0.0001):
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy')
    return model

def sample_action(policy_model, state):
    probabilities = policy_model.predict(state).flatten()
    return np.random.choice(len(probabilities), p=probabilities)

def compute_receding_window_average(rewards, window_size):
    avg_rewards = []
    for i in range(len(rewards)):
        if i < window_size:
            avg_rewards.append(np.mean(rewards[:i + 1]))
        else:
            avg_rewards.append(np.mean(rewards[i - window_size + 1:i + 1]))
    return avg_rewards

def train_policy_gradient(env, num_episodes, batch_size=32, learning_rate=0.0001):
    input_dim = 12
    output_dim = 20
    policy_model = create_policy_model(input_dim, output_dim, learning_rate)

    rewards_list = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        if obs is None or len(obs) == 0:
            print("Initial observation is empty. Exiting the episode.")
            break
        
        obs = np.concatenate(obs)
        num_zer = input_dim - obs.size
        if num_zer > 0:
            obs = np.pad(obs, (0, num_zer), mode='constant')
        obs = obs.reshape(1, -1)

        done = False
        # Updated section of the train_policy_gradient function
        while not done:
            # Sample action from policy
            action = sample_action(policy_model, obs)
        
            # Step in the environment
            obs_next, reward, _, done, _ = env.step((action // 10, action % 10))
        
            # Check if obs_next is empty
            if obs_next is None or len(obs_next) == 0:
                print("Next observation is empty. Exiting the loop.")
                break

            
            rewards_list.append(reward)
        
            # Move to the next state
            obs = obs_next
            
            # Check if obs is empty before concatenation
            if obs is None or len(obs) == 0:
                print("Observation is empty after step. Exiting the loop.")
                break
            
            # Check and concatenate the obs before reshaping
            if len(obs) > 0:  # Ensure there is something to concatenate
                obs = np.concatenate(obs)
                num_zer = input_dim - obs.size
                 
                if num_zer > 0:
                    obs = np.pad(obs, (0, num_zer), mode='constant')
                obs = obs.reshape(1, -1)
            else:
                print("Observation is empty after concatenation. Exiting the loop.")
                break

    return np.array(rewards_list)

# Main execution
if __name__ == "__main__":
    env = mmWaveEnv()
    num_episodes = 3  # Number of episodes for training

    # Run Linear Greedy
    n = math.log(20) / (2 * math.log(720))
    reward_lin_greedy = lin_greedy(env, 0.2, n)

    # Run Linear UCB
    reward_lin_ucb = lin_ucb(env, 0.2, n)

    # Run Policy Gradient
    reward_policy_gradient = train_policy_gradient(env, num_episodes)

    # Compute average rewards
    window_size = 100
    avg_reward_greedy = compute_receding_window_average(reward_lin_greedy, window_size)
    avg_reward_ucb = compute_receding_window_average(reward_lin_ucb, window_size)
    avg_reward_pg = compute_receding_window_average(reward_policy_gradient, window_size)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(avg_reward_greedy, label='Linear Greedy', alpha=0.7)
    plt.plot(avg_reward_ucb, label='Linear UCB', alpha=0.7)
    plt.plot(avg_reward_pg, label='Policy Gradient', alpha=0.7)
    
    plt.title('Receding Window Average Reward for Different Algorithms')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.axhline(y=np.mean(reward_lin_greedy), color='r', linestyle='--', label='Overall Avg Reward (Greedy)')
    plt.axhline(y=np.mean(reward_lin_ucb), color='g', linestyle='--', label='Overall Avg Reward (UCB)')
    plt.axhline(y=np.mean(reward_policy_gradient), color='b', linestyle='--', label='Overall Avg Reward (PG)')
    
    plt.legend()
    plt.grid()
    plt.savefig('combined_reward_plot.png')
    plt.show()

    env.close()
