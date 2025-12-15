import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

def entropy_func(theta):
    return np.sum(theta * np.log(np.clip(theta + 1e-12, 1e-12, 1)))
    # clipping is to ensure that the first arg(and array) has values that stay
    #in a certain range , here(between 1e-12 and 1),(between o and 1 not being zero)
    # entropy measures the uncertainity,(theta has the probabilites.. of what???)


def entropy_func_gradient(theta):
    return np.log(np.clip(theta + 1e-12, 1e-12, 1)) + 1
    # log of zero is undefined , so, the clipping

def prob_vector_generator(N, mu, stddev):
    bounds = Bounds(np.zeros(N + 1), np.ones(N + 1))
    
    A = np.zeros((3, N + 1))  # Updated to (3, N + 1) to match 3 constraints
    A[0, :] = 1
    A[1, :] = np.arange(N + 1)
    A[2, :] = np.arange(N + 1) ** 2
    
    b_lb = np.array([1, mu, mu ** 2 + stddev ** 2])
    b_ub = np.array([1, mu, mu ** 2 + stddev ** 2])
    #second moment is variance +mean square
    
    linear_constraint = LinearConstraint(A, b_lb, b_ub)
    #this ensures the three conditions:
        #sum of all thetas is 1
        #mean constraint:sum(i*theta_i)=mean(which is mu)
        #second moment constraint(sum((i^2)*theta_i)=(mu**2) + stddev**2)
        #following this , the code snippet , just initializes all theta as
        #1/(N+1), throught the minimize function , it is opimized so that
        #it satisfies all these properties
    
    x_start = (1 / (N + 1)) * np.ones(N + 1)
    result = minimize(
        entropy_func, 
        x_start, 
        method='trust-constr', 
        jac=entropy_func_gradient, 
        constraints=linear_constraint, 
        bounds=bounds
    )

    theta = np.clip(result.x, 0, 1)
    #this is just to ensure the results are in the right range
    theta = theta / np.sum(theta)    
    #normalisation...
    return theta  #GIVES A PROB DISTRIBUTION IN THE END....
D=5
# Generate arrival probability distribution, phi, of computational demand.
mu_d = 2                 # Mean of phi. You can vary this between 0.1*D to 0.9*D
                          # where D is the maximum computational demand.                          
stddev_ratio = 0.5        # You can vary this between 0.1 to 0.9. Higher value of
                          # stddev_ratio means a higher standard deviation of phi.                          

stddev_d = stddev_ratio*np.sqrt(D*(D-mu_d))     # Standard deviation of phi.
phi = prob_vector_generator(D, mu_d, stddev_d)  # Arrival probability distribution.

import matplotlib.pyplot as plt

# Plotting PMF
plt.bar(range(len(phi)), phi)
plt.title('Probability Mass Function')
plt.xlabel('Demand Level')
plt.ylabel('Probability')
plt.savefig('d_pmf.png')
plt.show()

# Plotting CDF
cdf = np.cumsum(phi)
plt.plot(range(len(cdf)), cdf)
plt.title('Cumulative Distribution Function')
plt.xlabel('Demand Level')
plt.ylabel('Cumulative Probability')
plt.savefig('d_cdf.png')
plt.show()

data = np.random.choice(np.arange(len(phi)), p=phi, size=10000)

# Plotting Histogram
plt.hist(data, bins=30, density=True, alpha=0.6, color='b')
plt.title('Histogram of Demand Distribution')
plt.xlabel('Demand Level')
plt.ylabel('Density')
plt.savefig('d_hist.png')
plt.show()

# Plotting Box Plot
plt.boxplot(data)
plt.title('Box Plot of Demand Distribution')
plt.ylabel('Demand Level')
plt.xticks([1], ['Demand'])
plt.savefig('d_box.png')
plt.show()

from scipy.stats import skew, kurtosis

# Calculate skewness and kurtosis
data_skewness = skew(data)
data_kurtosis = kurtosis(data)

print(f'Skewness: {data_skewness}')
print(f'Kurtosis: {data_kurtosis}')

# Calculate percentiles
percentiles = np.percentile(data, [25, 50, 75])  # 25th, 50th (median), 75th percentiles
print(f'25th Percentile: {percentiles[0]}')
print(f'50th Percentile (Median): {percentiles[1]}')
print(f'75th Percentile: {percentiles[2]}')

# Calculate specific quantiles (e.g., 0.1, 0.9)
quantiles = np.quantile(data, [0.1, 0.9])  # 10th and 90th quantiles
print(f'10th Quantile: {quantiles[0]}')
print(f'90th Quantile: {quantiles[1]}')



