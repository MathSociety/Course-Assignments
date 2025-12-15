import numpy as np
from Assignment2Tools import prob_vector_generator
from itertools import product

def policy_evaluation(iteration, D, tau, S_max, theta, alpha, phi, threshold, Kmin, beta, policy, Vinitial):
    # State space
    Ss = np.arange(S_max+1)
    Sd = np.arange(D+1)
    
    state_dim = [S_max+1]
    state_dim.extend([D+1]*(tau+1))
    
    # Value function initialization
    V = np.copy(Vinitial)
    V_new = np.copy(Vinitial)
    
    Delta = np.inf
    k = 1
    while Delta>threshold or k<=Kmin:
        
        for l in product(Sd, repeat=tau+1):
            for s in Ss:
                
                x = (s,) + l
                u, delta = policy[x]
                
                cur_delta = delta-l[0]
                l_dash = list(l[1:])
                for j in range(tau):
                    if cur_delta>l_dash[j]:
                        l_dash[j] = 0
                        cur_delta-=l_dash[j]
                    else:
                        l_dash[j] -= cur_delta
                        break
                        
                s_dash = [s+u,]
                x_dash_partial = tuple(s_dash+l_dash)
                V_new[x] = theta[0]*max(0,u)+ (theta[1]*(s+u) + theta[2]*delta**2) + sum(alpha*l[:-1]) + beta*sum(V[x_dash_partial]*phi)                        
                
        Delta = np.max(np.abs(V_new-V))
        V = np.copy(V_new)
            
        print(iteration, k, Delta)
        k+=1
         
    return V


def policy_iteration(D, tau, S_max, theta, alpha, phi, threshold, Kmin, beta):
    
    # State space
    Ss = np.arange(S_max+1)
    Sd = np.arange(D+1)
    
    state_dim = [S_max+1]
    state_dim.extend([D+1]*(tau+1))
    
    # Policy and value function initialization
    policy = np.zeros(state_dim+[2], dtype=int)
    for l in product(Sd, repeat=tau+1):
        for s in Ss:
            u = l[0] - s
            delta = l[0]
            policy[(s,)+l] = np.array([u, delta])
            
    policy_new = np.copy(policy)
    V = np.zeros(state_dim)
    
    converged = False
    iteration = 1
    while not(converged):
        # Policy evaluation
        V = policy_evaluation(iteration, D, tau, S_max, theta, alpha, phi, threshold, Kmin, beta, policy, V)
        
        # Policy update
        iteration+=1
        for l in product(Sd, repeat=tau+1):
            l_sum = sum(l)
            for s in Ss:
                
                u_min = l[0]-s
                u_max = max(0, min(l_sum, S_max)-s)
                min_Q = np.inf
                for u in range(u_min, u_max+1, 1):
                    for delta in range(l[0], s+u+1, 1):
                        
                        cur_delta = delta-l[0]
                        l_dash = list(l[1:])
                        for j in range(tau):
                            if cur_delta>l_dash[j]:
                                l_dash[j] = 0
                                cur_delta-=l_dash[j]
                            else:
                                l_dash[j] -= cur_delta
                                break
                        s_dash = [s+u,]
                        x_dash_partial = tuple(s_dash+l_dash)
                        Q = theta[0]*max(0,u)+ (theta[1]*(s+u) + theta[2]*delta**2) + sum(alpha*l[:-1]) + beta*sum(V[x_dash_partial]*phi)
                        
                        if Q<min_Q:
                            min_Q = Q
                            min_action = np.array([u, delta])
                            
                x = (s,) + l
                if min_Q<V[x]:
                    policy_new[x] = min_action
        
        converged = np.all(policy_new==policy)
        policy = np.copy(policy_new)
    

    # Finding value function of the final policy
    V = policy_evaluation(iteration, D, tau, S_max, theta, alpha, phi, threshold, Kmin, beta, policy, V)

    return V, policy


def value_iteration(D, tau, S_max, theta, alpha, phi, threshold, Kmin, beta):
    
    # State space
    Ss = np.arange(S_max+1)
    Sd = np.arange(D+1)
    
    state_dim = [S_max+1]
    state_dim.extend([D+1]*(tau+1))
    
    # Value function initialization
    V = np.zeros(state_dim)
    V_new = np.zeros(state_dim)
    
    Delta = np.inf
    k = 1
    
    # Finding the optimal value function
    while Delta>threshold or k<=Kmin:
        
        for l in product(Sd, repeat=tau+1):
            l_sum = sum(l)
            for s in Ss:
                
                u_min = l[0]-s
                u_max = max(0, min(l_sum, S_max)-s)
                min_Q = np.inf
                for u in range(u_min, u_max+1, 1):
                    for delta in range(l[0], s+u+1, 1):
                        cur_delta = delta-l[0]
                        l_dash = list(l[1:])
                        for j in range(tau):
                            if cur_delta>l_dash[j]:
                                l_dash[j] = 0
                                cur_delta-=l_dash[j]
                            else:
                                l_dash[j] -= cur_delta
                                break
                                
                        s_dash = [s+u,]
                        x_dash_partial = tuple(s_dash+l_dash)
                        Q = theta[0]*max(0,u)+ (theta[1]*(s+u) + theta[2]*delta**2) + sum(alpha*l[:-1]) + beta*sum(V[x_dash_partial]*phi)
                        
                        if Q<min_Q:
                            min_Q = Q
                            
                x = (s,) + l
                V_new[x] = min_Q
                        
                
        Delta = np.max(np.abs(V_new-V))
        V = np.copy(V_new)
            
        print(k, Delta)
        k+=1
        
        
    # Finding the optimal policy using optimal value function
    policy = np.zeros(state_dim+[2], dtype=int)
    for l in product(Sd, repeat=tau+1):
        l_sum = sum(l)
        for s in Ss:
            
            u_min = l[0]-s
            u_max = max(0, min(l_sum, S_max)-s)
            min_Q = np.inf
            for u in range(u_min, u_max+1, 1):
                for delta in range(l[0], s+u+1, 1):
                    
                    cur_delta = delta-l[0]
                    l_dash = list(l[1:])
                    for j in range(tau):
                        if cur_delta>l_dash[j]:
                            l_dash[j] = 0
                            cur_delta-=l_dash[j]
                        else:
                            l_dash[j] -= cur_delta
                            break
                    s_dash = [s+u,]
                    x_dash_partial = tuple(s_dash+l_dash)
                    Q = theta[0]*max(0,u)+ (theta[1]*(s+u) + theta[2]*delta**2) + sum(alpha*l[:-1]) + beta*sum(V[x_dash_partial]*phi)
                    
                    if Q<min_Q:
                        min_Q = Q
                        min_action = np.array([u, delta])
                        
            x = (s,) + l
            policy[x] = min_action
    

    # Finding value function of the final policy
    V = policy_evaluation(k, D, tau, S_max, theta, alpha, phi, threshold, Kmin, beta, policy, V)
  
    return V, policy


D = 5
tau = 4
S_max = 15


theta = np.array([10, 1, 0.2])
alpha = np.cumsum(0.3*np.ones(tau))


mu_d = 3.5
stddev_ratio = 0.7
stddev_d = stddev_ratio*np.sqrt(D*(D-mu_d))
phi = prob_vector_generator(D, mu_d, stddev_d)


threshold = 2
Kmin = 10
beta = 0.95


V_vi, policy_vi = value_iteration(D, tau, S_max, theta, alpha, phi, threshold, Kmin, beta)
V_pi, policy_pi = policy_iteration(D, tau, S_max, theta, alpha, phi, threshold, Kmin, beta)