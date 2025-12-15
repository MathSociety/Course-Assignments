import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils.play import play
import pygame

#lunar_dataset = pd.DataFrame(columns=['Play #','State', 'Action','Reward','Next State', 'Terminated'])

global total_reward
total_reward = 0

def save_play(state, next_state, action, reward, terminated, truncated, info):
    global lunar_dataset, total_reward
    
    # This if-else statement is used to calculate and display the total reward
    # of an episode.
    if terminated:
        total_reward+=reward
        print('Total reward in this episode is {}'.format(total_reward))
        total_reward = 0
    else:
        total_reward+=reward
        
    # This if-else statement is used to decide the "Play Number" (same as
    # episode number) of the current play.
    if len(lunar_dataset)==0:
        play_no = 1
    else:
        last_play_no = lunar_dataset.iloc[-1,0]
        last_terminated = lunar_dataset.iloc[-1,5]
        if last_terminated==1:
            play_no = last_play_no + 1
        else:
            play_no = last_play_no
    
    # This if statement is required only for t=0 because the environment
    # returns the current observation and a dictonary (info). We don't want
    # the dictonary.
    if len(state)==2:
        state = state[0] # Select only the state and reject the dictionary.
    
    # Append the new <x, a, r, x_dash, terminated> to existing dataset.
    df = {'Play #':[play_no], 'State':[state], 'Action':[action], 'Reward':[reward], 'Next State':[next_state], 'Terminated':[int(terminated)]}
    lunar_dataset = pd.concat([lunar_dataset,pd.DataFrame(df)], ignore_index=True)


# Load existing dataset
lunar_dataset = pd.read_csv('lunar_dataset.csv')

# Initiate the lunar lander environment
env = gym.make('LunarLander-v2', render_mode='rgb_array')

# Play the game
play(env, keys_to_action={'a': 1, 's': 2, 'd': 3}, noop=0, fps=10, callback=save_play)

# Close the environment
env.close()

# The following line needs to be executed to close the display incase the game
# was abruptly stopped.
# pygame.display.quit()

# Reject the last episode that started but was not played.
terminated_arr = np.array(lunar_dataset.iloc[:,-1]).astype(int)
N = np.max(np.arange(len(terminated_arr))[terminated_arr==1])
lunar_dataset = lunar_dataset.iloc[:N+1,:]

# Save the data
lunar_dataset.to_csv('lunar_dataset.csv', index=False)