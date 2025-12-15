import numpy as np
import random
from untitled2 import GymTrafficEnv


class TripleQLearningAgent:
    def __init__(self ,env ,alpha=0.1 ,beta=0.9 ,epsilon=0.1, n_actions=2):
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.n_actions = n_actions
        
        self.q_dict={
            1: np.zeros((21,21,n_actions,10,2)),
            2: np.zeros((21,21,n_actions,10,2)),
            3: np.zeros((21,21,n_actions,10,2))
        }
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,self.n_actions -1)
        else:
            q_values = sum(self.q_dict[q][state] for q in self.q_dict)
            return np.argmax(q_values)
        
    def update(self, state, action, reward, next_state):
        
        i=random.randint(1,3)
        U = {1,2,3} - {i}
        
        q_a,q_b=[self.q_dict[u][next_state][action] for u in U]
        q_next_avg = (q_a + q_b)/2

        self.q_dict[i][state][action] +=self.alpha * (
            reward + (self.beta)* q_next_avg - self.q_dict[i][state][action]) 
    
    def train(self,num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done=False
            
            while not done:
                action = self.choose_action(state)
                next_state,reward,done,_,_ = self.env.step(action)
                self.update(state,action,reward,next_state)
                state=next_state

if __name__ =="__main__":
    env= GymTrafficEnv()
    agent =TripleQLearningAgent(env)
    agent.train(num_episodes=1000)