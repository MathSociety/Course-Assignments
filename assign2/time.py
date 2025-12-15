import time

import numpy as np
from Assignment2Tools import prob_vector_generator
import itertools

def update_leftover_demands(state, delta_t, tau):
    """Update the leftover demands given state and delta_t."""
    d_t, *l_t, x_t = state
    l_t = list(l_t)  # Convert to a list for easy manipulation

    # Shift previous leftover demands and update the first leftover demand
    for i in range(tau - 1, 0, -1):
        l_t[i] = l_t[i - 1]
    l_t[0] = max(0, d_t - delta_t)  # Current demand not served

    return (d_t, *l_t, x_t)  # Return the new state

def reward_function(state, action, theta, alpha, tau):
    """Calculate the reward for a given state and action."""
    d_t, *l_t, x_t = state
    y_t, u_t = action
    immediate_reward = -theta[0] * max(u_t, 0) - theta[1] * x_t - theta[2] * (y_t ** 2)
    immediate_penalty = -sum(alpha[i] * l_t[i] for i in range(tau))  # Penalties for leftover demands
    return immediate_reward + immediate_penalty

def value_iteration(D, S_max, phi, theta, beta, tau, threshold=1e-6, Kmin=10):
    """Implement the Value Iteration algorithm."""
    state_space = list(itertools.product(range(D + 1), *[range(D + 1) for _ in range(tau)], range(S_max + 1)))
    value_function = np.zeros(len(state_space))
    delta = float('inf')
    iteration = 0

    while delta > threshold or iteration < Kmin:
        delta = 0  # Reset delta for the new iteration
        for idx, state in enumerate(state_space):
            v_new = float('-inf')  # Initialize new value

            # Possible actions based on state
            d_t, *l_t, x_t = state
            actions = [(y, u) for y in range(0, min(x_t, d_t + sum(l_t)) + 1)
                                for u in range(-x_t, S_max - x_t + 1)]

            for action in actions:
                y_t, u_t = action
                delta_t = min(y_t, d_t + sum(l_t))  # Calculate effective demand served

                # Calculate immediate reward
                immediate_reward = reward_function(state, action, theta, alpha, tau)

                # Calculate expected value for the next state
                updated_state = update_leftover_demands(state, delta_t, tau)
                expected_value = 0

                # Iterate through all possible next states based on updated demand
                for next_demand in range(D + 1):
                    next_state = (next_demand, *updated_state[1:tau + 1], updated_state[tau + 1])  # Create new state structure
                    if next_state in state_space:
                        prob = phi[next_demand]  # Probability associated with the next demand
                        expected_value += prob * value_function[state_space.index(next_state)]

                # Total value for the action
                total_value = immediate_reward + beta * expected_value
                v_new = max(v_new, total_value)

            # Update the value function for this state
            delta = max(delta, abs(value_function[idx] - v_new))
            value_function[idx] = v_new

        print('Iteration:', iteration)
        iteration += 1
    
    print('Value function computation done.')

    # Derive the optimal policy from the value function
    policy = np.zeros(len(state_space), dtype=object)
    for idx, state in enumerate(state_space):
        d_t, *l_t, x_t = state
        actions = [(y, u) for y in range(0, min(x_t, d_t + sum(l_t)) + 1)
                            for u in range(-x_t, S_max - x_t + 1)]
        best_action = None
        best_value = float('-inf')

        for action in actions:
            y_t, u_t = action
            delta_t = min(y_t, d_t + sum(l_t))  # Calculate effective demand served

            immediate_reward = reward_function(state, action, theta, alpha, tau)

            # Calculate expected value for the next state
            updated_state = update_leftover_demands(state, delta_t, tau)
            expected_value = 0
            for next_demand in range(D + 1):
                next_state = (next_demand, *updated_state[1:tau + 1], updated_state[tau + 1])
                if next_state in state_space:
                    prob = phi[next_demand]  # Probability associated with the next demand
                    expected_value += prob * value_function[state_space.index(next_state)]

            total_value = immediate_reward + beta * expected_value
            if total_value > best_value:
                best_value = total_value
                best_action = action

        policy[idx] = best_action  # Store the best action for this state
        print('Index:', idx)

    return value_function, policy

def policy_iteration(D, S_max, phi, theta, beta, tau, threshold=1e-6):
    """Implement the Policy Iteration algorithm."""
    state_space = list(itertools.product(range(D + 1), *[range(D + 1) for _ in range(tau)], range(S_max + 1)))
    num_states = len(state_space)
    state_indices = {state: i for i, state in enumerate(state_space)}  # Precompute state indices

    # Initialize policy arbitrarily
    policy = [(0, 0) for _ in range(num_states)]
    value_function = np.zeros(num_states)

    converged = False
    while not converged:
        # Step 3: Policy Evaluation
        value_function = policy_evaluation(policy, D, S_max, phi, theta, beta, tau)

        converged = True
        # Step 4: Policy Improvement
        for idx, state in enumerate(state_space):
            d_t, *l_t, x_t = state
            actions = [(y, u) for y in range(0, min(x_t, d_t + sum(l_t)) + 1)
                                for u in range(-x_t, S_max - x_t + 1)]
            best_action = policy[idx]
            best_value = float('-inf')

            for action in actions:
                immediate_reward = reward_function(state, action, theta, alpha, tau)
                delta_t = min(action[0], d_t + sum(l_t))
                
                # Calculate expected value based on next possible demands
                expected_value = 0
                for next_demand in range(D + 1):
                    prob = phi[next_demand]
                    updated_state = (next_demand, *update_leftover_demands(state, delta_t, tau)[1:])
                    next_state_idx = state_indices.get(updated_state)
                    if next_state_idx is not None:
                        expected_value += prob * value_function[next_state_idx]

                total_value = immediate_reward + beta * expected_value

                if total_value > best_value:
                    best_value = total_value
                    best_action = action

            if best_action != policy[idx]:
                policy[idx] = best_action
                converged = False

    return policy, value_function


def measure_computation_time(algorithm, *args):
    start_time = time.time()
    algorithm(*args)
    end_time = time.time()
    return end_time - start_time

# System parameters (set to default values)
D = 5                            # Maximum computational demand
tau = 4                          # Maximum deferrable time.
S_max = 15                       # Maximum number of servers that can be on at any time.
theta = np.array([10, 1, 0.2])   # Parameters associated with server cost.
alpha = np.cumsum(0.3 * np.ones(tau))  # Parameters associated with penalty cost.

# Generate arrival probability distribution, phi, of computational demand.
mu_d = 2                  # Mean of phi. You can vary this between 0.1*D to 0.9*D
stddev_ratio = 0.5        # You can vary this between 0.1 to 0.9. Higher value of
                          # stddev_ratio means a higher standard deviation of phi.                          

stddev_d = stddev_ratio * np.sqrt(D * (D - mu_d))  # Standard deviation of phi.
phi = prob_vector_generator(D, mu_d, stddev_d)  # Arrival probability distribution.

beta = 0.95  # Discount factor

# Convergence parameters
threshold = 2   # The absolute value of the difference between current and 
                # updated value function FOR ANY STATE should be lesser than
                # this threshold for convergence.                
Kmin = 10       # Minimum number of iterations of value iteration.

# Assuming value_iteration and policy_iteration are defined as per your earlier codes.
value_times = []
policy_times = []

for _ in range(5):
    # Generate new phi for each run
    phi = prob_vector_generator(D, mu_d, stddev_d)  
    value_time = measure_computation_time(value_iteration, D, S_max, phi, theta, beta, tau, threshold, Kmin)
    policy_time = measure_computation_time(policy_iteration, D, S_max, phi, theta, beta, tau)
    
    value_times.append(value_time)
    policy_times.append(policy_time)

average_value_time = sum(value_times) / len(value_times)
average_policy_time = sum(policy_times) / len(policy_times)

print(f"Average computation time for value iteration: {average_value_time:.4f} seconds")
print(f"Average computation time for policy iteration: {average_policy_time:.4f} seconds")
