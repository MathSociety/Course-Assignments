import numpy as np
import itertools
from Assignment2Tools import prob_vector_generator

def policy_evaluation(theta,D,S_max,beta,Kmin,threshold):
    # You must decide the necessary arguments of policy_evaluation.
    # This function must return the value function of the greedy policy.
    # pass
    
    def policy(median,skewness,state,action,D):#depends on the distribution (phi)
        dt=state[0]
        xt=state[1]
        if skewness < -0.17: #(demands are more on the higher side)###############
            if xt==dt:#--------------------------
                if action==1:
                    return 1
                else:
                    return 0
            elif xt<dt:#-------------------------
                if xt >= median:
                    if action == (dt-xt):
                        return 1
                    else:
                        return 0
                elif xt <median:
                    if action = (dt-xt) + 1:
                        return 1
                    else:
                        return 0
            elif xt>dt:#-------------------------
                if xt >=median:
                    if action ==0:
                        return 1
                    else:
                        return 0
                elif xt<median:
                    if action ==1:
                        return 1
                    else:
                        return 0
        elif skewness >0.17: #(demands are more on the lower side)#################
            if xt ==dt:#-------------------------
                if action==0:
                    return 0
                else:
                    return 1
            elif xt<dt:#-------------------------
                if xt>=median:
                    if action ==(dt=xt):
                        return 1
                    else:
                        return 0
                elif xt<medium:
                    if action ==(dt-xt)+1:
                        return 1
                    else:
                        return 0
            elif xt>dt:#--------------------------
                if xt>median:
                    if action == -1:
                        return 1
                    else:
                        return 0
                elif xt<=median:
                    if action ==0:
                        return 1
                    else:
                        return 0
        elif skewness<=0.17 and skewness>=-0.17:#################################
            if xt==dt:
                if xt>=median:
                    if action==0:
                        return 1
                    else:
                        return 0
                elif xt<median:
                    if action ==1:
                        return 1
                    else:
                        return 0
            elif xt<dt:
                if action ==dt-xt:
                    return 1
                else:
                    return 0
            elif xt>dt:
                if x>medium:
                    if action ==-1:
                        return 1
                    else:
                        return 0
                elif xt<=median:
                    if action ==1:
                        return 1
                    else:
                        return 
    def action_space(state,Smax):
        dt=state[0]
        xt=state[1]
        actionspace=list(itertools.product(dt,range(dt-xt,S_max -xt+1)))
        return actionspace
        
    state_space=list(itertools.product(range(D+1),range(S_max+1)))
    value_function=np.zeros(len(state_space))
    value_function[:]=0;
    new_value_function=np.zeros(len(state_space))
    new_value_function[:]=0;
    del=float('inf')
    
    
    while del> theta:
        for state in state_space:
            new_value_function[state_space.index(state)]=
    
        
        
    return value_function;


# System parameters (set to default values)
D = 5                            # Maximum computational demand
S_max = 15                       # Maximum number of servers that can be on at any time.
theta = np.array([10, 1, 0.2])   # Parameters associated with server cost.
                                 # theta_1 = 10, theta_2 = 1, and theta_3 = 0.2.


# Generate arrival probability distribution, phi, of computational demand.
mu_d = 2                  # Mean of phi. You can vary this between 0.1*D to 0.9*D
                          # where D is the maximum computational demand.                          
stddev_ratio = 0.5        # You can vary this between 0.1 to 0.9. Higher value of
                          # stddev_ratio means a higher standard deviation of phi.                          

stddev_d = stddev_ratio*np.sqrt(D*(D-mu_d))     # Standard deviation of phi.
phi = prob_vector_generator(D, mu_d, stddev_d)  # Arrival probability distribution.

# getting the 50th quantile (or the median) and the skewness for the greedy policy 
# this is using properties of the demand distribution to say, for example if the demands are skewed toward 
# zero,and on servers are greater than the demads --> we will be turning off one computer.
data = np.random.choice(np.arange(len(phi)), p=phi, size=1000000)
skewness = skew(data)
median = np.percentile(data, [50])

beta = 0.95 # Discount factor

# Convergence parameters
threshold = 2   # The absolute value of the difference between current and 
                # updated value function FOR ANY STATE should be lesser than
                # this threshold for convergence.                
Kmin = 10       # Minimum number of iterations of iterative policy evaluation.


# Call policy_evaluation function. You must decide the necessary arguments of policy_evaluation.
V = policy_evaluation()
