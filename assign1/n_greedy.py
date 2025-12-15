import gymnasium as gym
import numpy as np

from mmWave_bandits import mmWaveEnv

def lin_greedy(env):
    
    Naction=20
    Nfeatures=12
    delta=0.02
    theta= np.zeros((Naction,Nfeatures+1))
    A=np.zeros((Naction,Nfeatures+1,Nfeatures+1))
    I=np.eye(A.shape[0])
    A=A+ (delta*I)
    b= np.zeros((Naction,Nfeatures+1))
    epsilon =0.1
    
    Nepisodes = 10
    
    reward_arr=[]
    
    for _ in range(Nepisodes):
        t=0
        
        obs,_ = env.reset()
        
        z = np.concatenate(obs).reshape(-1,1)
        z=np.pad(z,((0,12-z.shape[0]),(0,0)),'constant',constant_values=0)
        z= np.concatenate((z,np.array([[1]])),axis=0)
        
        truncated =False
        while not(truncated):
            
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
            theta[action] = np.matmul(np.linalg.inv(A[action] + 0.01*np.eye(Nfeatures+1), b[action]))
            
            obs= obs_next
            z = np.concatenate(obs).reshape(-1,1)
            z=np.pad(z,((0,12-z.shape[0]),(0,0)),'constant',constant_values=0)
            z= np.concatenate((z,np.array([[1]])),axis=0)
            t+=1
            
    return np.array(reward_arr), theta


env=mmWaveEnv()
reward, theta= lin_greedy(env)
env.close()


















