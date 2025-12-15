import numpy as np
from GymTraffic import GymTrafficEnv
import random

def TripleQLearning(env, beta, Nepisodes, alpha):
    epsilon = 0.1
    n_actions = 2
            
    q_array = np.zeros((3, 21, 21, n_actions, 10, 2))
        
    def choose_action(state):
        if random.random() < epsilon:
            return random.randint(0, n_actions - 1)
        else:
            q_values_sum = np.sum(q_array[:,state[0],state[1],:,state[3],state[2]], axis=0)
            return np.argmax(q_values_sum)
            
    def update(state, action, reward, next_state):
        i = random.randint(0, 2)
        U = {0, 1,2} - {i}
            
        q_a, q_b = [q_array[u,next_state[0],next_state[1],action,next_state[3],next_state[2]] for u in U]
        q_next_avg = (q_a + q_b) / 2

        q_array[i,state[0],state[1],action,state[3],state[2]] += alpha * (
            reward + (beta) * q_next_avg - q_array[i ,state[0] , state[1], action, state[3], state[2]]
        )
        
    for episode in range(Nepisodes):
        state = list(env.reset())
        state[0] = min(state[0],20)
        state[1] = min(state[1],20)
        state=tuple(state)
        done = False
                
        print(f"Episode: {episode +1}")
        while not done:
            action = choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = list(next_state)
            next_state[0] = min(next_state[0],20)
            next_state[1] = min(next_state[1],20)
            next_state=tuple(next_state)
            update(state, action, reward, next_state)
            
            state = next_state
            env.render()
    return q_array


def ModifiedSARSA(env, beta, Nepisodes, alpha):
    epsilon = 0.1
        
    Q = np.zeros((21,21,2,10,2))
        
    def choose_action(state):
        Q1, Q2, G, T = state
        Q1 = min(Q1,20)
        Q2 = min(Q2,20)
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[Q1, Q2, G, T])
        
    for episode in range(Nepisodes):
        state = env.reset()
        state=list(state)
        state[0] = min(state[0],20)
        state[1] = min(state[1],20)
        state=tuple(state)    
        G = []
        done = False
        
        print(f"Episode: {episode +1}")
            
        action = choose_action(state)
            
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_state = list(next_state)
            next_state[0] = min(next_state[0],20)
            next_state[1] = min(next_state[1],20)
            next_state = tuple(next_state)
            G.append(reward)
                
            next_action = choose_action(next_state)
                
            if len(G) == Nepisodes + 1:
                update_value = sum(((beta) ** i) * G[i] for i in range(Nepisodes)) + \
                               ((beta) ** (Nepisodes + 1)) * Q[next_state][next_action]
                Q[state][action] += alpha * (update_value - Q[state][action])
                G.pop(0)
                
            state = next_state
            action = next_action
            env.render()
            if done:
                break
                
        G.clear()
        
    return Q


env = GymTrafficEnv()  # Create an instance of the traffic controller environment.

Nepisodes = 10000  # Number of episodes to train
alpha = 0.1        # Learning rate
beta = 0.997       # Discount factor

# Learn the optimal policies using two different TD learning approaches
policy1 = TripleQLearning(env, beta, Nepisodes, alpha)
policy2 = ModifiedSARSA(env, beta, Nepisodes, alpha)

# Save the policies
np.save('TripleQLearning_Policy.npy', policy1)
np.save('ModifiedSARSA_Policy.npy', policy2)

env.close()