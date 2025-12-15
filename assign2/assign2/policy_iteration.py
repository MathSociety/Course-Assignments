import numpy as np
from Assignment2Tools import prob_vector_generator
import itertools

def policy_evaluation(policy, D, S_max, phi, theta, beta, tau, threshold=1e-6):
    """Evaluate the value function for a given policy."""
    if tau == 0:
        state_space = list(itertools.product(range(D + 1), range(S_max + 1)))
    else:
        state_space = list(itertools.product(range(D + 1), *[range(D + 1) for _ in range(tau)], range(S_max + 1)))

    num_states = len(state_space)
    value_function = np.zeros(num_states)
    state_indices = {state: i for i, state in enumerate(state_space)}  

    while True:
        delta = 0
        new_values = np.zeros(num_states)  
        for idx, state in enumerate(state_space):
            action = policy[idx]
            immediate_reward = reward_function(state, action, theta, alpha, tau)

            
            if tau == 0:
                expected_value = sum(
                    phi[next_demand] * value_function[state_indices.get((next_demand, action[1]), 0)]
                    for next_demand in range(D + 1)
                )
            else:
                expected_value = sum(
                    phi[next_demand] * value_function[state_indices.get((
                        next_demand, *update_leftover_demands(state, min(action[0], state[0] + sum(state[1:tau + 1])), tau)[1:])
                    , 0)]
                    for next_demand in range(D + 1)
                )


            new_values[idx] = immediate_reward + beta * expected_value
            delta = max(delta, abs(value_function[idx] - new_values[idx]))

        value_function = new_values  

        if delta < threshold:
            break

    return value_function

def policy_iteration(D, S_max, phi, theta, beta, tau, threshold=2):
    """Implement the Policy Iteration algorithm."""
    if tau == 0:
        state_space = list(itertools.product(range(D + 1), range(S_max + 1)))
    else:
        state_space = list(itertools.product(range(D + 1), *[range(D + 1) for _ in range(tau)], range(S_max + 1)))

    num_states = len(state_space)
    state_indices = {state: i for i, state in enumerate(state_space)}

    policy = [(0, 0) for _ in range(num_states)]
    value_function = np.zeros(num_states)

    converged = False
    iteration_count = 0
    while not converged:
        value_function = policy_evaluation(policy, D, S_max, phi, theta, beta, tau)

        converged = True
        for idx, state in enumerate(state_space):
            d_t, *l_t, x_t = state
            actions = [(y, u) for y in range(0, min(x_t, d_t + sum(l_t)) + 1)
                                for u in range(max(0, -x_t), S_max - x_t + 1)] 
            best_value = float('-inf')
            current_action = policy[idx]

            for action in actions:
                immediate_reward = reward_function(state, action, theta, alpha, tau)
                delta_t = min(action[0], d_t + sum(l_t))

                if tau == 0:
                    expected_value = sum(
                        phi[next_demand] * value_function[state_indices.get((next_demand, action[1]), 0)]
                        for next_demand in range(D + 1)
                    )
                else:
                    expected_value = sum(
                        phi[next_demand] * value_function[state_indices.get((
                            next_demand, *update_leftover_demands(state, delta_t, tau)[1:]), 0)]
                        for next_demand in range(D + 1)
                    )

                total_value = immediate_reward + beta * expected_value
                if total_value > best_value:
                    best_value = total_value
                    best_action = action

            if best_action != current_action:
                policy[idx] = best_action
                converged = False

        iteration_count += 1 
        print(f"Iteration {iteration_count} completed.")

    return policy, value_function

def reward_function(state, action, theta, alpha, tau):
    """Calculate the reward for a given state and action."""
    d_t, *l_t, x_t = state
    y_t, u_t = action
    immediate_reward = -theta[0] * max(u_t, 0) - theta[1] * x_t - theta[2] * (y_t ** 2)
    immediate_penalty = -sum(alpha[i] * l_t[i] for i in range(tau)) if tau > 0 else 0
    return immediate_reward + immediate_penalty

def update_leftover_demands(state, delta_t, tau):
    """Update the leftover demands given state and delta_t."""
    d_t, *l_t, x_t = state
    l_t = list(l_t)
    if tau > 0:
        for i in range(tau - 1, 0, -1):
            l_t[i] = l_t[i - 1]
        l_t[0] = max(0, d_t - delta_t)
    return (d_t, *l_t, x_t)

# System parameters
D = 5
tau = 4  # Change this to 0 for testing
S_max = 15
theta = np.array([10, 1, 0.2])
alpha = np.cumsum(0.3 * np.ones(tau)) if tau > 0 else np.zeros(0)  # Avoids issue if tau = 0
mu_d = 2
stddev_ratio = 0.5
stddev_d = stddev_ratio * np.sqrt(D * (D - mu_d))
phi = prob_vector_generator(D, mu_d, stddev_d)
beta = 0.95
threshold = 2
Kmin = 10

# Run policy iteration
policy_optimal_pi, V_optimal_pi = policy_iteration(D, S_max, phi, theta, beta, tau)


print("Optimal Policy:", policy_optimal_pi)
print("Optimal Value Function:", V_optimal_pi)
