'''
Created on Dec 15, 2014

@author: Francois Belletti
'''

from cvxpy import *
import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt

DEFAULT_N_ITERATIONS = 25
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0

def get_u_n(l, m, p, mu, T, sigma):
    x = Variable(T)
    objective = Minimize((np.atleast_2d(mu) * x) + sigma * sum_squares(x))
    constraints = [l <= p * x, p * x <= m]
    prob = Problem(objective, constraints)
    result = prob.solve(solver = CVXOPT)
    return x.value
    
def get_gradient(lower_matrix_list,
                 upper_matrix_list,
                 core_matrix_list,
                 mu, o_vect,
                 N, T, sigma,
                 noise_level = 0.0):
    grad = -0.5 * mu - o_vect
    obj = -0.25 * np.sum((mu ** 2)) - np.dot(mu, o_vect)
    for i in range(N):
        temp = np.ravel(get_u_n(lower_matrix_list[i],
                            upper_matrix_list[i], 
                            core_matrix_list[i], 
                            mu * 
                            (1.0 + noise_level * np.random.normal(0.0, 0.1, len(mu))),
                            T, sigma))
        temp *= (1.0 + noise_level * np.random.normal(0.0, 1.0, (len(temp))))
        obj += np.dot(mu, temp) + sigma * np.sum(temp ** 2)
        grad += temp
    return grad, obj

def solve(lower_matrix_list,
          upper_matrix_list,
          core_matrix_list,
          o_vect,
          N, T, sigma,
          A,
          noise_level = 0.0):
    #
    #    mu is lambda here (dual optimial price vector)
    #    it is initialized at random
    #
    mu = np.random.normal(0.0, 1.0, T)
    #
    obj_values = []
    alpha = DEFAULT_ALPHA
    #
    #    Iterate through gradient ascent
    #
    print 'DOCSP running'
    for i in range(DEFAULT_N_ITERATIONS):
        temp, obj = get_gradient(lower_matrix_list,
                                 upper_matrix_list,
                                 core_matrix_list,
                                 mu, o_vect,
                                 N, T, sigma, 
                                 noise_level)
        obj_values.append(obj)
        mu += alpha / np.power(i + 1, DEFAULT_BETA) * temp
        #print obj
        #print mu
        cost_vector = mu
    print 'Done'
    #
    #    Finalizing step
    #
    x_matrix = np.zeros((N, T))
    for i in range(N):
        x_matrix[i] = np.ravel(get_u_n(lower_matrix_list[i], 
                                         upper_matrix_list[i],
                                         core_matrix_list[i], 
                                         cost_vector,
                                         T, sigma))
    solution = np.ravel(x_matrix)
    obj_value = np.sum((np.dot(A, solution) - o_vect) ** 2) + sigma * np.sum(solution ** 2)
    return solution, obj_value, cost_vector, obj_values
    