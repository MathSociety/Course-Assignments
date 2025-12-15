import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from GymTraffic import GymTrafficEnv

def TestPolicy(env, policy,policy_name):
    env = GymTrafficEnv()
    
    q_len_road1 = []
    q_len_road2 = []
    time_steps_over =0
    state = env.reset()
    Q1,Q2,G,T=state
    Q1=min(Q1,20)
    Q2=min(Q2,20)

        
    done = False
    if policy.shape[0] == 3:
        q_values_sum = sum(policy[i][Q1,Q2,G,T] for i in range(3))
        action = np.argmax(q_values_sum)
    else:
        action = np.argmax(policy[Q1,Q2,G,T])  
    
    
    while not done:
        
        Q1, Q2, _, _ = state
        if Q1>20:
            Q1=20
        if Q2>20:
            Q2=20
            
        q_len_road1.append(Q1)
        q_len_road2.append(Q2)
        
        if policy.shape[0] ==3:
            q_values_sum = sum(policy[i][Q1,Q2,G,T] for i in range(3))
            action = np.argmax(q_values_sum)
        else:
            action = np.argmax(policy[Q1,Q2,G,T])     
        next_state, reward, done, _,_ = env.step(action)
        state = next_state
        time_steps_over += 1
        
    avg_q_len = np.mean(np.array(q_len_road1) + np.array(q_len_road2))
    
    time_steps = np.arange(time_steps_over)
    plt.figure(figsize=(12,6))
    plt.plot(time_steps,q_len_road1,label='queue length road 1',color='blue')
    plt.plot(time_steps,q_len_road2,label='queue length road 2',color='red')
    plt.title('Queue lengths of roads over time')
    plt.xlabel('time steps')
    plt.ylabel('queue length')
    plt.legend()
    
    print(f"average queue length over one episode: {avg_q_len} ,method: {policy_name} ")
    plt.savefig(f'road_queue_length_{policy_name}.png')
    
    plt.figure()
    plt.plot(time_steps,np.array(q_len_road1) + np.array(q_len_road2) ,label="total queue length",color='black')
    plt.title("Both road's total queue length")
    plt.xlabel("time_steps")
    plt.ylabel("total queue length")
    plt.legend()
    plt.savefig(f'Total_length_{policy_name}.png')
    
    
    
env = GymTrafficEnv() # Create and instance of the traffic controller environment.

#policy = np.load('TripleQLearning_Policy.npy',allow_pickle=True) # To load Triple Q-Learning policy.
policy = np.load('ModifiedSARSA_Policy.npy') # To load modified SARSA policy.

#TestPolicy(env, policy,policy_name="TripleQLearning")
TestPolicy(env, policy,policy_name="ModifiedSARSA")

env.close() # Close the environment