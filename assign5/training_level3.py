import numpy as np
import gymnasium as gym
import pygame
from cartpole_continuous import CartPoleEnv_Continuous
from stable_baselines3 import PPO
from torch import nn  # Importing nn for activation functions

# Custom Wrapper to modify rewards based on proximity to a setpoint
class Custom_Wrapper(gym.Wrapper):
    def __init__(self, env, setpoint, reward_scale=1.0):
        super().__init__(env)
        self.setpoint = setpoint  # Desired position
        self.reward_scale = reward_scale  # Scaling factor for reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Assuming observation[0] is the cart position (you may need to adjust this depending on your observation structure)
        cart_position = observation[0]  
        
        # Calculate the distance from the setpoint
        distance_from_setpoint = abs(cart_position - self.setpoint)
        
        # Custom reward function: Reward for getting closer to the setpoint
        if distance_from_setpoint < 0.1:
            modified_reward = 1.0  # Reward for reaching close to setpoint
        elif distance_from_setpoint < 0.2:
            modified_reward = 0.5  # Small reward for approaching the setpoint
        else:
            modified_reward = -0.1  # Penalty for moving away from the setpoint
        
        modified_reward *= self.reward_scale  # Apply reward scaling factor
        
        return observation, modified_reward, terminated, truncated, info

def evaluate_model_performance(model, Nframes, control_mode, mode):
    if mode == 'analyze':
        env = CartPoleEnv_Continuous(control_mode=control_mode)
        Nepisodes = 100
    elif mode == 'visualize':
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
        
        if mode == 'visualize':
            print('Total reward = {}'.format(total_reward))
    
    env.close()
    
    if mode == 'analyze':
        avg_reward = np.mean(total_reward_arr)
        print('Average of total reward = {}'.format(avg_reward))


control_mode = 'setpoint'
setpoint = 0.0  
env = CartPoleEnv_Continuous(control_mode=control_mode)


env = Custom_Wrapper(env, setpoint, reward_scale=1.0)

Nframes = 1 


policy_kwargs = dict(
    net_arch=[64, 64],  
    activation_fn=nn.Tanh  
)


ppo_model = PPO(
    "MlpPolicy",  
    env,
    learning_rate=1e-3,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,
    vf_coef=0.5,
    clip_range=0.2,
    policy_kwargs=policy_kwargs, 
    verbose=1
)
ppo_model.learn(total_timesteps=50000, log_interval=4)


evaluate_model_performance(ppo_model, Nframes, control_mode, 'analyze')


evaluate_model_performance(ppo_model, Nframes, control_mode, 'visualize')


env.close()
