'''
Created on Dec 15, 2014

@author: Francois Belletti
'''

import numpy as np

def compute_regularized_obj(A, solution, o_vect, sigma):
    return np.sum((np.dot(A, solution) - o_vect) ** 2) + sigma * np.sum(solution ** 2)

def compute_original_obj(A, solution, o_vect, sigma):
    return np.sum((np.dot(A, solution) - o_vect) ** 2)