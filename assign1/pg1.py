import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from mmWave_bandits import mmWaveEnv

# Define the policy model using Keras
def create_policy_network(context_dim, action_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(context_dim,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(action_dim, activation='softmax'))  # Softmax for action probabilities
    return model

# Sample an action from the policy
def get_action(policy_model, context):
    context = tf.convert_to_tensor(context, dtype=tf.float32)  # Ensure it's a tensor
    prob = policy_model(context)[0].numpy()  # Run forward pass
    action = np.random.choice(len(prob), p=prob)
    return action, prob

# Define the Policy Gradient loss
def policy_gradient_loss(probabilities, action, reward):
    action_prob = probabilities[action] + 1e-10  # Add epsilon to avoid log(0)
    loss = -tf.math.log(action_prob) * reward  # Policy Gradient loss
    return loss

# Training loop for Policy Gradient
def train_policy_gradient(env, num_episodes, batch_size=32, learning_rate=0.0001):
    context_dim = 12  # Adjust based on your context size
    action_dim = 20   # Adjust based on your action space size
    policy_model = create_policy_network(context_dim, action_dim)
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    all_rewards = []

    for episode in range(num_episodes):
        context, _ = env.reset()  # Get the initial context

        # Ensure context is not empty before concatenation
        if context is None or len(context) == 0:
            print("Initial observation is empty. Exiting the episode.")
            break

        context = np.concatenate(context)
        num_zeros = context_dim - context.size

        if num_zeros > 0:
            context = np.pad(context, (0, num_zeros), mode='constant')
        context = context.reshape(1, -1)

        done = False
        episode_rewards = []

        while not done:
            with tf.GradientTape() as tape:
                action, probabilities = get_action(policy_model, context)
                next_context, reward, _, truncated, _ = env.step((action // 10, action % 10))
                done = truncated

                # Calculate loss
                loss = policy_gradient_loss(probabilities, action, reward)

            # Update policy network using gradients
            grads = tape.gradient(loss, policy_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

            context = next_context
            # Check if context is empty before concatenation
            if context is None or len(context) == 0:
                print("Context is empty after step. Exiting the loop.")
                break

            context = np.concatenate(context)
            num_zeros = context_dim - context.size

            if num_zeros > 0:
                context = np.pad(context, (0, num_zeros), mode='constant')
            context = context.reshape(1, -1)

            episode_rewards.append(reward)

        total_reward = np.sum(episode_rewards)
        all_rewards.append(total_reward)
        print(f'Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}')

    return all_rewards, policy_model

# Compute receding window average
def compute_receding_window_average(rewards, window_size):
    avg_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    return avg_rewards

if __name__ == "__main__":
    env = mmWaveEnv()
    num_episodes = 80  # Number of episodes for training
    batch_size = 32  # Set batch size
    learning_rate = 0.0001  # Set learning rate

    # Train the policy gradient agent
    rewards, trained_policy = train_policy_gradient(env, num_episodes, batch_size=batch_size, learning_rate=learning_rate)

    # Compute and plot receding window average reward
    window_size = 100  # Adjust window size if needed
    avg_rewards = compute_receding_window_average(rewards, window_size)

    plt.plot(avg_rewards, label='Receding Window Avg Reward')
    plt.title('Policy Gradient: Receding Window Average Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.axhline(y=np.mean(rewards), color='r', linestyle='--', label='Overall Average Reward')
    plt.legend()
    plt.grid()
    plt.savefig('policy_gradient_rewards.png')
    plt.show()

    env.close()
