import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math

from mmWave_bandits import mmWaveEnv

def lin_greedy(env,d,n):
    
    Naction=20
    Ncontext=12
    theta= np.zeros((Naction,Ncontext+1))
    A=np.zeros((Naction,Ncontext+1,Ncontext+1))
    b= np.zeros((Naction,Ncontext+1))
    
    
    Nepisodes = 10
    
    reward_arr=[]
    
    for _ in range(Nepisodes):
        t = 0
        obs, _ = env.reset()
        
        z = np.concatenate(obs).reshape(-1, 1)
        z = np.pad(z, ((0, 12 - z.shape[0]), (0, 0)), 'constant', constant_values=0)
        z = np.concatenate((z, np.array([[1]])), axis=0)
        
        truncated = False
        while t<720 and not truncated:
            epsilon = max(d / (t + 1) ** n, 0.01)
            v = np.random.uniform(0, 1)
        
            if v <= epsilon:
                action = np.random.randint(Naction)
            else:
                action = np.argmax(np.matmul(theta, z))
        
            obs_next, reward, _, truncated, _ = env.step((action // 10, action % 10))
            reward_arr.append(reward)
            print(reward)
        
            if not truncated:
                z_temp = z.reshape((-1, 1))
                A[action] += np.matmul(z_temp, np.transpose(z_temp))
                b[action] += reward * z_temp.reshape(-1)
                theta[action] = np.matmul(np.linalg.inv(A[action] + 0.01 * np.eye(Ncontext + 1)), b[action])
        
                obs = obs_next
                z = np.concatenate(obs).reshape(-1, 1)
                z = np.pad(z, ((0, 12 - z.shape[0]), (0, 0)), 'constant', constant_values=0)
                z = np.concatenate((z, np.array([[1]])), axis=0)
            
            t += 1  # Move this to the end of the loop

            
    return np.array(reward_arr), theta


env=mmWaveEnv()
#  (initial epsilon value/720^n)>0.01 (min epsilon value)
# so n<log (20) base 720  =0.45533
# lesser n --> slower decay of epsilon
# larger n --> faster decay of epsilon

n = math.log(20) / (2* math.log(720))
reward, theta= lin_greedy(env,0.2,n)
env.close()
window_size=100
avg_reward=[]
for i in range(len(reward)):
    if i<window_size:
        avg_reward.append(np.mean( reward[:i+1]) )
    else:
        avg_reward.append(np.mean( reward[i-window_size+1 :i+1]) )
print(avg_reward[0])
plt.plot(avg_reward,label='receding window avg reward')
plt.title('lin greedy')
plt.xlabel('Time Steps')
plt.ylabel('Average Reward')
plt.axhline(y=np.mean(reward), color='r', linestyle='--', label='Overall Average Reward')
plt.legend()
plt.grid()
plt.savefig('lin_greedy.png')
plt.show()


















