import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def Q_func(h, c, a, V, N, tau, f_arr, p_arr, beta):
    if c==0 and a==0:
        val = f_arr[h-1] + beta*(p_arr[h-1]*V[h-1,0] + (1-p_arr[h-1])*V[max(1,h-1)-1,0])
    elif c==0 and a==1:
        val = beta*V[h-1,tau-1]
    elif c==1:
        val = beta*V[N-1,0]
    else:
        val = beta*V[h-1,c-1]
    
    return val

def policy_evaluation(N, tau, f_arr, p_arr, theta, Kmin, beta, policy, Vinitial):
    # State space
    S_h = np.arange(1,N+1)   # State space for the health state
    S_c = np.arange(tau)     # State space for the counter state

    # Value function initialization
    V = np.copy(Vinitial)
    V_new = np.zeros((N, tau))

    Delta = np.inf # Delta holds the difference between old and new value of V. Used to decide convergence.
    k = 1
    while Delta>theta or k<=Kmin: # k<Kmin is not there in the psuedocode for iterative policy ealuation. This additional condition just means that a minimum number of iteration must happen.
        # Step S3 and S4 of the psuedocode of iterative policy evaluation.
        for h, c in product(S_h, S_c):
            a = policy[h-1,c]            
            V_new[h-1, c] = Q_func(h, c, a, V, N, tau, f_arr, p_arr, beta)
        
        # Step S5 of the psuedocode of iterative policy evaluation.
        Delta = np.max(np.abs(V_new-V))
        
        # Step S6 of the psuedocode of iterative policy evaluation.
        V = np.copy(V_new)
        
        k+=1
        
    return V

def policy_iteration(N, tau, f_arr, p_arr, theta, Kmin, beta):
    # State space
    S_h = np.arange(1,N+1)   # State space for the health state
    S_c = np.arange(tau)     # State space for the counter state
    
    # Policy initialization (Policy must me initialized such that the initialized action for a state is in its action space.)
    policy = np.zeros((N, tau), dtype=int)
    policy[:,1:tau] = -1
    
    policy_new  = np.copy(policy) # This will be used to store the update policy in every iteration.
    
    # Value function initialization
    V = np.zeros((N, tau))
    
    converged = False
    while not(converged):
        # Policy evaluation (Step S3 of the psuedocode of policy iteration)
        V = policy_evaluation(N, tau, f_arr, p_arr, theta, Kmin, beta, policy, V)
        
        # Policy updation (Step S4 of the psuedocode of policy iteration)
        for h, c in product(S_h, S_c):
            
            # (Step S5 of the psuedocode of policy iteration)
            if c>0:
                policy_new[h-1,c] = policy[h-1,c] # c>0 implies that the machine is in repair.
            else:
                Q0 = Q_func(h, c, 0, V, N, tau, f_arr, p_arr, beta) # Q function for NOT repairing
                Q1 = Q_func(h, c, 1, V, N, tau, f_arr, p_arr, beta) # Q function for repairing
                Qmax = np.max([Q0, Q1])
                
                # (Step S6 of the psuedocode of policy iteration)
                if Qmax>V[h-1,c]:
                    policy_new[h-1,c] = np.argmax([Q0, Q1])
                else:
                    policy_new[h-1,c] = policy[h-1,c]
        
        # Step S7 of the psuedocode of policy iteration.This step checks if any of the policy value has changed.
        converged = np.all(policy_new==policy)
        
        # Step S8 of the psuedocode of policy iteration. Sets the policy for the next iteration as the updated policy.
        policy = np.copy(policy_new)
        
    return policy
                

# System parameters
N = 10     # Number of health state
tau = 5    # Repair time
f_arr = np.cumsum(1+np.random.uniform(size=(N))) # Monetary value of the product. cumsum ensures f(h) is monotonic increasing.
ph = 0.8  # Probability of NOT degrading to a lower health state. 
p_arr = ph*np.ones(N)

# Convergence parameters for iterative policy evaluation
theta = 0.001
Kmin = 10

beta = 0.99 # Discount factor
policy = policy_iteration(N, tau, f_arr, p_arr, theta, Kmin, beta)

# Visualize the policy
plt.figure(1)
plt.stem(np.arange(1,N+1,1), policy[:,0]) # Only plotting policy when c=0, i.e. machine not in repair state.
plt.xlabel('Health State h (Worst --> Best)', fontsize=14)
plt.ylabel('Action a', fontsize=14)
plt.xticks(np.arange(1,N+1,1), fontsize=12)
plt.yticks(np.arange(0,2,1), fontsize=12)