import numpy as np
from Assignment2Tools import prob_vector_generator
import itertools

def update_leftover_demands(state, delta_t, tau):
    """Update the leftover demands given state and delta_t."""
    d_t, *l_t, x_t = state
    if tau == 0:
        return (d_t, 0, x_t) 
    l_t = list(l_t) 
    l_t = [max(0, l_t[i - 1]) for i in range(tau)] 
    l_t[0] = max(0, d_t - delta_t) 
    return (d_t, *l_t, x_t) 

def reward_function(state, action, theta, alpha, tau):
    """Calculate the reward for a given state and action."""
    d_t, *l_t, x_t = state
    y_t, u_t = action
    immediate_reward = -theta[0] * max(u_t, 0) - theta[1] * x_t - theta[2] * (y_t ** 2)
    
    immediate_penalty = -sum(alpha[i] * l_t[i] for i in range(tau)) if tau > 0 else 0
    
    return immediate_reward + immediate_penalty

def value_iteration(D, S_max, phi, theta, beta, tau, threshold=2, Kmin=10):
    """Implement the Value Iteration algorithm."""
    state_space = list(itertools.product(range(D + 1), *[range(D + 1) for _ in range(tau)], range(S_max + 1)))
    state_indices = {state: i for i, state in enumerate(state_space)}
    num_states = len(state_space)
    value_function = np.zeros(num_states)

    delta = float('inf')
    iteration = 0
    while delta > threshold or iteration < Kmin:
        delta = 0
        new_values = np.zeros(num_states) 

        for idx, state in enumerate(state_space):
            d_t, *l_t, x_t = state
            actions = [(y, u) for y in range(0, min(x_t, d_t + sum(l_t)) + 1)
                                for u in range(-x_t, S_max - x_t + 1)]
            best_value = float('-inf')

            for action in actions:
                y_t, u_t = action
                delta_t = min(y_t, d_t + sum(l_t))
                immediate_reward = reward_function(state, action, theta, alpha, tau)

                updated_state = update_leftover_demands(state, delta_t, tau)
                expected_value = sum(
                    phi[next_demand] * value_function[state_indices.get(
                        (next_demand, *updated_state[1:tau + 1], updated_state[tau + 1])
                    )] for next_demand in range(D + 1)
                )

                total_value = immediate_reward + beta * expected_value
                best_value = max(best_value, total_value)

            new_values[idx] = best_value
            delta = max(delta, abs(value_function[idx] - best_value))

        value_function = new_values
        print(f'Iteration {iteration}, max delta {delta}')
        iteration += 1

    policy = np.zeros(len(state_space), dtype=object)
    for idx, state in enumerate(state_space):
        d_t, *l_t, x_t = state
        actions = [(y, u) for y in range(0, min(x_t, d_t + sum(l_t)) + 1)
                            for u in range(-x_t, S_max - x_t + 1)]
        best_action = None
        best_value = float('-inf')

        for action in actions:
            y_t, u_t = action
            delta_t = min(y_t, d_t + sum(l_t))
            immediate_reward = reward_function(state, action, theta, alpha, tau)
            updated_state = update_leftover_demands(state, delta_t, tau)
            expected_value = sum(
                phi[next_demand] * value_function[state_indices.get(
                    (next_demand, *updated_state[1:tau + 1], updated_state[tau + 1])
                )] for next_demand in range(D + 1)
            )

            total_value = immediate_reward + beta * expected_value
            if total_value > best_value:
                best_value = total_value
                best_action = action

        policy[idx] = best_action 
        print(f'Policy optimized for state index {idx}')

    return value_function, policy

D = 5
tau = 4  
S_max = 15
mu_d = 1.0    
stddev_ratio = 0.2 
stddev_d = stddev_ratio * np.sqrt(D * (D - mu_d)) 
phi = prob_vector_generator(D, mu_d, stddev_d) 
theta = np.array([10, 1, 0.2])
alpha = np.cumsum(0.3 * np.ones(tau)) if tau > 0 else np.array([0]) 
beta = 0.95 
threshold =2
Kmin = 10     



V_optimal, policy_optimal = value_iteration(D, S_max, phi, theta, beta, tau)


print("Optimal Value Function:", V_optimal)
print("Optimal Policy:", policy_optimal)
