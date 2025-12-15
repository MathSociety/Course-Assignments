import numpy as np
from Assignment2Tools import prob_vector_generator
from itertools import product

def policy_evaluation(iteration , D, tau, S_max, theta,alpha, phi, threshold, Kmin,beta,policy,Vinitial):
     Ss = np.arange(S_max+1)
     Sd = np.arange(D+1)
     
     state_dim = [S_max+1]
     state_dim.extend([D+1]*(tau+1))
     
     V = np.cpoy(Vinitial)
     V_new = np.copy(Vinitial)
     
     Delta = np.inf
     k=1
     while Delta>threshhold