import numpy as np
from Assignment2Tools import prob_vector_generator
import itertools

def update_leftover_demands(state, delta_t, tau):
    """Update the leftover demands given state and delta_t."""
    d_t, *l_t, x_t = state
    if tau == 0:
        return (d_t, 0, x_t)  # No leftover demands when tau is 0
    l_t = list(l_t)  # Convert to list for manipulation
    l_t = [max(0, l_t[i - 1]) for i in range(tau)]  # Shift demands down
    l_t[0] = max(0, d_t - delta_t)  # Update with unserved current demand
    return (d_t, *l_t, x_t)  # Return the new state

def reward_function(state, action, theta, alpha, tau):
    """Calculate the reward for a given state and action."""
    d_t, *l_t, x_t = state
    y_t, u_t = action
    immediate_reward = -theta[0] * max(u_t, 0) - theta[1] * x_t - theta[2] * (y_t ** 2)
    
    # When tau is 0, there are no leftover penalties
    immediate_penalty = -sum(alpha[i] * l_t[i] for i in range(tau)) if tau > 0 else 0
    
    return immediate_reward + immediate_penalty

def value_iteration(D, S_max, phi, theta, beta, tau, threshold=1e-6, Kmin=10):
    """Implement the Value Iteration algorithm."""
    state_space = list(itertools.product(range(D + 1), *[range(D + 1) for _ in range(tau)], range(S_max + 1)))
    state_indices = {state: i for i, state in enumerate(state_space)}
    num_states = len(state_space)
    value_function = np.zeros(num_states)

    delta = float('inf')
    iteration = 0
    while delta > threshold or iteration < Kmin:
        delta = 0
        new_values = np.zeros(num_states)  # Temporary array for updated values

        for idx, state in enumerate(state_space):
            d_t, *l_t, x_t = state
            actions = [(y, u) for y in range(0, min(x_t, d_t + sum(l_t)) + 1)
                                for u in range(-x_t, S_max - x_t + 1)]
            best_value = float('-inf')

            for action in actions:
                y_t, u_t = action
                delta_t = min(y_t, d_t + sum(l_t))
                immediate_reward = reward_function(state, action, theta, alpha, tau)

                # Calculate expected value for the next state
                updated_state = update_leftover_demands(state, delta_t, tau)
                expected_value = sum(
                    phi[next_demand] * value_function[state_indices.get(
                        (next_demand, *updated_state[1:tau + 1], updated_state[tau + 1])
                    )] for next_demand in range(D + 1)
                )

                total_value = immediate_reward + beta * expected_value
                best_value = max(best_value, total_value)

            new_values[idx] = best_value  # Store the best value for the state
            delta = max(delta, abs(value_function[idx] - best_value))

        value_function = new_values  # Update value function in one step for this iteration
        print(f'Iteration {iteration}, max delta {delta}')
        iteration += 1

    # Derive the optimal policy based on the computed value function
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

        policy[idx] = best_action  # Best action for this state
        print(f'Policy optimized for state index {idx}')

    return value_function, policy

# Smaller System Parameters
D = 5
tau = 4  # Change this to 0 for testing
S_max = 15

# Generate arrival probability distribution, phi, of computational demand.
mu_d = 1.0     # Mean of phi
stddev_ratio = 0.2  # Lower standard deviation
stddev_d = stddev_ratio * np.sqrt(D * (D - mu_d))  # Standard deviation of phi.
phi = prob_vector_generator(D, mu_d, stddev_d)  # Arrival probability distribution.
theta = np.array([10, 1, 0.2])
alpha = np.cumsum(0.3 * np.ones(tau)) if tau > 0 else np.array([0])  # Ensure alpha is initialized properly
beta = 0.95  # Discount factor
threshold = 2  # Tighter threshold for convergence
Kmin = 10       # Minimum number of iterations of value iteration.

import time
import csv


# Initialize time records
value_iteration_times = []

# Open CSV file for writing results
with open('value_iteration_times.csv', mode='w', newline='') as csvfile:
    fieldnames = ['Run', 'mu_d', 'stddev_d', 'Value Iteration Time', 'Policy Iteration Time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()

    # Run experiments
    num_runs = 5
    for run in range(num_runs):
        # Generate a different phi for each run
        mu_d = np.random.uniform(0, D)  # Random mean demand
        stddev_ratio = np.random.uniform(0.1, 0.5)  # Random stddev ratio
        stddev_d = stddev_ratio * np.sqrt(D * (D - mu_d))
        phi = prob_vector_generator(D, mu_d, stddev_d)

        # Measure time for value iteration
        start_time = time.time()
        value_function = value_iteration(D, S_max, phi, theta, beta, tau)
        value_iteration_time = time.time() - start_time
        value_iteration_times.append(value_iteration_time)

        # Write results for the current run to CSV
        writer.writerow({
            'Run': run + 1,
            'mu_d': mu_d,
            'stddev_d': stddev_d,
            'Value Iteration Time': value_iteration_time
        })

# Calculate average times
avg_value_iteration_time = np.mean(value_iteration_times)

# Output average results
print(f"Average Value Iteration Time: {avg_value_iteration_time:.4f} seconds")


