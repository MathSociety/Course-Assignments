import numpy as np
import gymnasium as gym
import pygame

from stable_baselines3 import DQN

def visualize_model_performance(model):
    env = gym.make('LunarLander-v2', render_mode='human')
    terminated, truncated = False, False
    x, _ = env.reset()
    t = 0
    total_reward = 0
    while not(terminated) and not(truncated):
        action, _ = model.predict(x)
        x, reward, terminated, truncated, _ = env.step(action)        
        total_reward += reward
        t+=1
        if t==1000:
            truncated = True
    
    print('Total reward = {}'.format(reward))
    env.close()
    # pygame.display.quit() # Use this line whe the display screen is not going away
        


env = gym.make('LunarLander-v2')




model = DQN('MlpPolicy', env, verbose=1, train_freq=5, learning_rate=1e-3, 
            learning_starts=1000, batch_size=64, target_update_interval=10000, 
            buffer_size=50000, exploration_initial_eps=1.0, exploration_final_eps=0.1,
            policy_kwargs=dict(net_arch=[64, 64]))
model.learn(total_timesteps=200000)



env.close



visualize_model_performance(model)
