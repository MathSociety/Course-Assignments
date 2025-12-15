import itertools
import numpy as np
from Assignment2Tools import prob_vector_generator

def non_uniform_policy(d_t, x_t, S_max):
    # Example policy: prioritize actions based on demand compared to current servers
    actions = range(- (x_t - d_t), S_max - x_t + 1) if d_t < x_t else range(S_max - x_t + 1)
    
    # Create a probability distribution over actions
    pi = np.zeros(len(actions))
    
    for i, u_t in enumerate(actions):
        if d_t > x_t:
            pi[i] = 1  # Favor turning on more servers
        elif d_t < x_t:
            pi[i] = 0.5  # Lesser probability for turning off servers
        else:
            pi[i] = 0.2  # Low probability for no change

    pi /= pi.sum()  # Normalize to ensure probabilities sum to 1
    return pi

def policy_evaluation(D, S_max, phi, theta, beta, threshold=2, Kmin=10):
    # Initialize value function
    state_space = list(itertools.product(range(D + 1), range(S_max + 1)))
    value_function = np.zeros(len(state_space))
    
    # Initialize delta to infinity
    delta = float('inf')
    iteration = 0

    while delta > threshold or iteration < Kmin:
        delta = 0  # Reset delta for the new iteration
        for idx, (d_t, x_t) in enumerate(state_space):
            # Initialize the new value for the current state
            v_new = 0

            # Possible actions based on the state
            actions = range(- (x_t - d_t), S_max - x_t + 1) if d_t < x_t else range(S_max - x_t + 1)

            # Get action probabilities from the non-uniform policy
            action_probabilities = non_uniform_policy(d_t, x_t, S_max)

            for i, u_t in enumerate(actions):
                # Calculate the immediate reward
                immediate_reward = -theta[0] * max(u_t, 0) - theta[1] * x_t - theta[2] * (d_t ** 2)

                # Calculate the expected value for the next state
                expected_value = 0
                for d_prime in range(D + 1):
                    for x_prime in range(S_max + 1):
                        prob = phi[d_prime]  # Probability distribution over d_t
                        expected_value += prob * value_function[state_space.index((d_prime, x_prime))]
                
                # Total value for the action
                total_value = immediate_reward + beta * expected_value

                # Use the probability for the current action
                pi_s_a = action_probabilities[i]  # Probability for this action
                v_new += pi_s_a * total_value  # Weight total value by the policy

            # Update the value function for this state
            value_function[idx] = v_new
            
            # Calculate delta for convergence check
            delta = max(delta, abs(value_function[idx] - v_new))
        
        iteration += 1
    
    return value_function
