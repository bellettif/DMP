'''
Created on Dec 15, 2014

@author: Francois Belletti
'''

from cvxpy import *
import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt

def solve(L_matrix,
          M_matrix,
          P_matrix,
          o_vect,
          N, T, sigma,
          A):
    x = Variable(N*T)
    objective = Minimize(sum_squares(A * x - o_vect) + sigma * sum_squares(x))
    constraints = [L_matrix <= P_matrix * x, 
                   P_matrix * x <= M_matrix]
    prob = Problem(objective, constraints)    
    result = prob.solve(solver = CVXOPT)
    solution = np.ravel(np.asarray(x.value))
    obj_value = np.sum((np.dot(A, solution) - o_vect) ** 2) + sigma * np.sum(solution ** 2)
    return solution, obj_value

def solve_no_reg(L_matrix,
          M_matrix,
          P_matrix,
          o_vect,
          N, T,
          A):
    x = Variable(N*T)
    objective = Minimize(sum_squares(A * x - o_vect))
    constraints = [L_matrix <= P_matrix * x, 
                   P_matrix * x <= M_matrix]
    prob = Problem(objective, constraints)    
    result = prob.solve(solver = CVXOPT)
    solution = np.ravel(np.asarray(x.value))
    obj_value = np.sum((np.dot(A, solution) - o_vect) ** 2)
    return solution, obj_value