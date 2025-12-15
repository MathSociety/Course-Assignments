import numpy as np
import gymnasium as gym
import pygame
from cartpole_continuous import CartPoleEnv_Continuous

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
def evaluate_model_performance(model, Nframes, control_mode, mode):
    # If mode=='analyze', this function will calculate the average of the total reward over 100 episodes.
    # If mode=='visualize', this function will animate an episode.    
    
    if mode=='analyze':
        env = CartPoleEnv_Continuous(control_mode=control_mode)
        Nepisodes = 100
    elif mode=='visualize':
        env = CartPoleEnv_Continuous(control_mode=control_mode, render_mode='human')
        Nepisodes = 1
    else:
        print('Incorrect mode')
        return -1
    
    env = gym.wrappers.FrameStack(env, Nframes)
    total_reward_arr = []
    for e in range(Nepisodes):
        terminated, truncated = False, False
        observation, _ = env.reset()
        total_reward = 0
        while not(terminated) and not(truncated):
            action, _ = model.predict(observation)
            observation, reward, terminated, truncated, _ = env.step(action[0])        
            total_reward += reward
        
        total_reward_arr.append(total_reward)
        
        if mode=='visualize':
            print('Total reward = {}'.format(total_reward))
    
    env.close()
    # pygame.display.quit() # Use this line whe the display screen is not going away
    
    if mode=='analyze':
        avg_reward = np.mean(total_reward_arr)
        print('Average of total reward = {}'.format(avg_reward))
    


control_mode = 'regulatory'  
env = CartPoleEnv_Continuous(control_mode=control_mode)


Nframes = 10  # Set Nframes to 1, 2, 4, 6, 8, and 10. For value of Nframe, train the model and evaluate the performance.
env = gym.wrappers.FrameStack(env, Nframes)


n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))


model = DDPG(
    "MlpPolicy", 
    env, 
    policy_kwargs={"net_arch": [64, 64]},  # Network architecture with a max of 64 units per layer
    action_noise=action_noise,
    learning_rate=1e-3,  # Set learning rate
    buffer_size=1000000,  # Set buffer size
    learning_starts=100,  # Set learning starts
    batch_size=64,  # Set batch size
    tau=0.005,  # Soft update coefficient
    gamma=0.99,  # Discount factor
    train_freq=(1, "episode"),  # Update every episode
    gradient_steps=1,  # Perform 1 gradient step per rollout
    verbose=1
)
model.learn(total_timesteps=50000, log_interval=4)


env.close()


evaluate_model_performance(model, Nframes, control_mode, 'analyze')


evaluate_model_performance(model, Nframes, control_mode, 'visualize')
