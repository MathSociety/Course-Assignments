import numpy as np
import gymnasium as gym
from untitled2 import GymTrafficEnv

class ModifiedSARSA:
    def __init__(self, env, n, alpha=0.1, beta =0.9,epsilon=0.1):
        self.env = env
        self.n = n
        self.alpha =alpha
        self.beta = beta
        self.epsilon = epsilon
        
        self.Q = np.zeros((21,21,2,10,2))
        
    def choose_action(self,state):
        Q1, Q2, G, T = state
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[Q1,Q2,G,T])
        
    def train(self,episodes):
        for episode in range(episodes):
            state = self.env.reset()
            G=[]
            
            action = self.choose_action(state)
            
            while True:
                next_state, reward, done, _, _=self.env.step(action)
                G.append(reward)
                
                next_action = self.choose_action(next_state)
                
                if len(G) == self.n +1:
                    update_value= sum(((self.beta)**i) * G[i] for i in range(self.n)) +\
                        ((self.beta)**(self.n +1)) * self.Q[next_state][next_action]
                    self.Q[state][action] +=self.alpha *(update_value - self.Q[state][action])    
                    G.pop(0)
                
                state = next_state
                action = next_action
            
                if done:
                    break
                
            G.clear()
            
env=GymTrafficEnv()
n=1
agent = ModifiedSARSA(env, n)
agent.train(episodes=1000)