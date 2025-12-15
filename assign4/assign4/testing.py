import numpy as np
import gymnasium as gym
import pygame
import tensorflow as tf

def choose_action(state, model):
    """
    Choose an action based on the DQN model and the current state.
    Exploration is not required during testing.
    
    Args:
        state: Current state of the environment.
        model: Trained DQN model.

    Returns:
        int: Chosen action.
    """
    action_values = [
        model.predict([np.expand_dims(state, axis=0), np.array([[action]])], verbose=0)[0][0]
        for action in range(env.action_space.n)
    ]
    return np.argmax(action_values)


# The following line loads the DQN model.
model = tf.keras.models.load_model('lunar_lander_model.keras')

# The following line initializes the Lunar Lander environment with render_mode set to 'human'.
env = gym.make('LunarLander-v2', render_mode='human')

# The following line resets the environment.
state, _ = env.reset()

end_episode = False
total_reward = 0

while not end_episode:
    # The following line picks an action using choose_action() function.
    action = choose_action(state, model)

    # The following line takes the picked action. After taking the action, it gets
    # next state, reward, terminated, truncated, and info.
    next_state, reward, terminated, truncated, info = env.step(action)

    # The following line updates the total reward.
    total_reward += reward

    # The following line decides the state for the next time slot.
    state = next_state

    # The following line decides end_episode for the next time slot.
    end_episode = terminated or truncated

# The following line prints the total reward.
print(f"Total Reward: {total_reward}")

# The following line closes the environment.
env.close()

pygame.display.quit()
