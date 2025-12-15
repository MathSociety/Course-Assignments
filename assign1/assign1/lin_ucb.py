import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math
from mmWave_bandits import mmWaveEnv

def lin_ucb(env,d,n):
    
    Naction=20
    Nfeatures=12
    delta=0.02
    theta= np.zeros((Naction,Nfeatures+1))
    A=np.zeros((Naction,Nfeatures+1,Nfeatures+1))
    I=np.eye(Nfeatures+1)
    for i in range(Naction):
        A[i] += delta*I
    b= np.zeros((Naction,Nfeatures+1))
    
    
    Nepisodes = 10
    
    reward_arr=[]
    
    for _ in range(Nepisodes):
        t=0
        
        obs,_ = env.reset()
        
        z = np.concatenate(obs).reshape(-1,1)
        z=np.pad(z,((0,12-z.shape[0]),(0,0)),'constant',constant_values=0)
        z= np.concatenate((z,np.array([[1]])),axis=0)
        
        truncated =False
        while t<720 and not(truncated):
            epsilon = max(d / (t + 1) ** n, 0.01)
            values=[]
            for i in range(Naction):
                A_inv= np.linalg.inv(A[i])
                
                firstpart=z.T @ theta[i].reshape(-1,1)
                second= z.T @ A_inv @ z
                sqrt_term= np.sqrt(second[0,0])
                value= firstpart[0,0] + (epsilon* sqrt_term)
                values.append(value)
                
            action=np.argmax(values)
                
            obs_next,reward,_,truncated, _=env.step((action//10,action%10))
            reward_arr.append(reward)
            print(reward)
        
            if not(truncated):
                z_temp = z.reshape((-1,1))
                A[action] +=np.matmul(z_temp,np.transpose(z_temp))
                b[action] +=reward*z_temp.reshape(-1)
                theta[action] = np.matmul(np.linalg.inv(A[action] + 0.01 * np.eye(Nfeatures + 1)), b[action])

                
                obs= obs_next
                z = np.concatenate(obs).reshape(-1,1)
                z=np.pad(z,((0,12-z.shape[0]),(0,0)),'constant',constant_values=0)
                z= np.concatenate((z,np.array([[1]])),axis=0)
            t+=1
            
    return np.array(reward_arr), theta


env=mmWaveEnv()
#  (initial epsilon value/720^n)>0.01 (min epsilon value)
# so n<log (20) base 720  =0.45533
# lesser n --> slower decay of epsilon
# larger n --> faster decay of epsilon

n = math.log(20) / (2* math.log(720))
reward, theta= lin_ucb(env,0.2,n)

env.close()
window_size=100
avg_reward=[]
for i in range(len(reward)):
    if i<window_size:
        avg_reward.append(np.mean( reward[:i+1]) )
    else:
        avg_reward.append(np.mean( reward[i-window_size+1 :i+1]) )

plt.plot(avg_reward,label='receding window avg reward')
plt.title('lin ucb')
plt.xlabel('Time Steps')
plt.ylabel('Average Reward')
plt.axhline(y=np.mean(reward), color='r', linestyle='--', label='Overall Average Reward')
plt.legend()
plt.grid()
plt.savefig('lin_ucbplot.png')
plt.show()


















