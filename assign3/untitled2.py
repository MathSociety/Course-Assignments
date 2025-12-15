import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GymTrafficEnv(gym.Env):
    def __init__(self):
        self.max_queue_length=20
        self.arrival_prob_road1=0.245
        self.arrival_prob_road2=0.35
        self.p_1=0.2
        self.p_h=0.9
        self.tau=10
        self.time_limit=1800
        
        self.observation_space=spaces.Tuple((
            spaces.Discrete(self.max_queue_length +1),
            spaces.Discrete(self.max_queue_length +1),
            spaces.Discrete(2),
            spaces.Discrete(10)))
        
        self.action_space=spaces.Discrete(2)
        self.reset()
    
    def reset(self):
        
        self.Q1=np.random.randint(0,11)
        self.Q2=np.random.randint(0,11)
        self.G=np.random.choice([0,1])
        self.T=0
        self.current_time=0
        return (self.Q1,self.Q2,self.G,self.T)
    
    def step(self,action):
        
        if action ==1:
            self.G= 1-self.G
            self.T=0
        else:
            self.T= min(self.T +1,9)
            
        if np.random.rand() < self.arrival_prob_road1:
            self.Q1 = min(self.Q1 + 1,self.max_queue_length)
        if np.random.rand() <self.arrival_prob_road2:
            self.Q2 = min(self.Q2 + 1,self.max_queue_length)
            
        dep_prob= self.p_1 +(self.p_h -self.p_1)*min(self.T,self.tau)/self.tau

        if self.G == 0:
            if np.random.rand() < dep_prob:
                self.Q1 = max(self.Q1 -1,0)
        else:
            if np.random.rand() <dep_prob:
                self.Q2=max(self.Q2 -1,0)
                
        reward= -(self.Q1 +self.Q2)
        
        self.current_time+=1
        terminated = self.current_time >=self.time_limit
        truncated = False
        
        return(self.Q1, self.Q2 ,self.G, self.T), reward, terminated, truncated,{}
    
    def render(self):
        print(f"Q1: {self.Q1}, Q2: {self.Q2}, G: {self.G}, T: {self.T}")
            
            
env= GymTrafficEnv()            
state=env.reset()

for _ in range(1800):
    action = env.action_space.sample()
    state,reward,terminated,truncated,info=env.step(action)
    env.render()
    
    if terminated:
        break
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        