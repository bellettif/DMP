'''
Created on Dec 15, 2014

@author: Francois Belletti
'''

import numpy as np
from matplotlib import pyplot as plt
import time
import cPickle as pickle

import primal_solver, dual_solver
import utils

# Problem data.
T = 50
N = 5
sigma = 1.0
NOISE_LEVEL_DOCSP = 0.0

clock_measurements = {}
error_measurements = {}

lower_matrix_list = []
core_matrix_list = []
upper_matrix_list = []
for i in range(N):
    lower_matrix_list.append(-np.ones((2 * T, 1)) * 40
                             #+ np.random.normal(0.0, 1.0, (2 * T, 1))
                             )
    core_matrix_list.append(np.vstack((np.tril(np.ones((T, T))),
                                       np.eye(T))))
    upper_matrix_list.append(np.ones((2 * T, 1)) * 10
                             #+ np.random.normal(0.0, 1.0, (2 * T, 1))
                             )

L_matrix = np.vstack(lower_matrix_list)
M_matrix = np.vstack(upper_matrix_list)

P_matrix = np.zeros((2 * N * T, N * T))
for i in range(N):
    P_matrix[i * 2 * T: (i + 1) * 2 * T, i * T : (i + 1) * T] = core_matrix_list[i]
    
A = np.zeros((T, N * T))
for i in range(T):
    A[i, i::T] = np.ones(N)
    
d = 3 + np.linspace(0, 8, T) ** 2
o_vect = -d

#
#    Solve original problem with CVX
#
start = time.clock()
solution_original, obj_value_original = primal_solver.solve_no_reg(L_matrix, 
                                                                   M_matrix, 
                                                                   P_matrix, 
                                                                   o_vect, 
                                                                   N, T, A)
stop = time.clock()
time_taken = stop - start
print 'Original problem, reconstruction error = %.2f, took %.2fs' % (obj_value_original, time_taken)
plt.title('Original, reconstruction error = %.2f' % obj_value_original)
plt.plot(d)
plt.plot(- np.dot(A, solution_original))
plt.legend(('d', 'sum_u'), 'upper left')
plt.savefig('Original_T=%d_N=%d_sigma=%.2f.png' % (T, N, sigma), 
            dpi = 300)
plt.close()

clock_measurements['original'] = time_taken
error_measurements['original'] = obj_value_original


#
#    Solve regularized problem with CVX
#
start = time.clock()
solution_primal, obj_value_primal = primal_solver.solve(L_matrix, 
                                                        M_matrix, 
                                                        P_matrix, 
                                                        o_vect, 
                                                        N, T, 
                                                        sigma, 
                                                        A)
stop = time.clock()
time_taken = stop - start
reconstruction_error_primal = utils.compute_original_obj(A, 
                                                         solution_primal,
                                                         o_vect,
                                                         sigma)
print 'Regularized problem, reconstruction error = %.2f, time taken = %.2f' % \
            (reconstruction_error_primal, time_taken)
plt.title('Primal regularized, reconstruction error = %.2f' % reconstruction_error_primal)
plt.plot(d)
plt.plot(- np.dot(A, solution_primal))
plt.legend(('d', 'sum_u'), 'upper left')
plt.savefig('Primal_T=%d_N=%d_sigma=%.2f.png' % (T, N, sigma), dpi = 300)
plt.close()

clock_measurements['primal'] = time_taken
error_measurements['primal'] = reconstruction_error_primal

#
#    Solve with DOCSP
#
start = time.clock()
solution_dual, obj_value_dual, cost_vector_dual, obj_values_dual = \
        dual_solver.solve(lower_matrix_list, 
                          upper_matrix_list, 
                          core_matrix_list, 
                          o_vect,
                          N, T, sigma, A,
                          noise_level = NOISE_LEVEL_DOCSP)
stop = time.clock()
time_taken = stop - start
reconstruction_error_DOCSP = utils.compute_original_obj(A, 
                                                        solution_dual,
                                                        o_vect,
                                                        sigma)
plt.title('Control convergence DOCSP')
plt.plot(obj_values_dual)
plt.xlabel('Iteration')
plt.ylabel('Obj value')
plt.savefig('DOCSP_conv_T=%d_N=%d_sigma=%.2f_noise=%.2f.png' % (T, N, sigma, NOISE_LEVEL_DOCSP),
            dpi = 300)
plt.close()
print 'DOCSP problem, reconstruction error = %.2f, time taken = %.2f' %\
        (reconstruction_error_DOCSP, time_taken)
plt.title('DOCSP, reconstruction error = %.2f' % reconstruction_error_DOCSP)
plt.plot(d)
plt.plot(-np.dot(A, solution_dual))
plt.plot(cost_vector_dual)
plt.legend(('d', 'sum_u', 'cost vector'), 'upper left')
plt.savefig('DOCSP_T=%d_N=%d_sigma=%.2f_noise=%.2f.png' % (T, N, sigma, NOISE_LEVEL_DOCSP),
            dpi = 300)
plt.close()

clock_measurements['DOCSP'] = time_taken
error_measurements['DOCSP'] = reconstruction_error_DOCSP

#
#    Solve with DOCSP (low noise in com)
#
NOISE_LEVEL_DOCSP = 0.1
start = time.clock()
solution_dual, obj_value_dual, cost_vector_dual, obj_values_dual = \
        dual_solver.solve(lower_matrix_list, 
                          upper_matrix_list, 
                          core_matrix_list, 
                          o_vect,
                          N, T, sigma, A,
                          noise_level = NOISE_LEVEL_DOCSP)
stop = time.clock()
time_taken = stop - start
plt.title('Control convergence DOCSP')
plt.plot(obj_values_dual)
plt.xlabel('Iteration')
plt.ylabel('Obj value')
plt.savefig('DOCSP_conv_T=%d_N=%d_sigma=%.2f_noise=%.2f.png' % (T, N, sigma, NOISE_LEVEL_DOCSP),
            dpi = 300)
plt.close()
reconstruction_error_DOCSP = utils.compute_original_obj(A, 
                                                        solution_dual,
                                                        o_vect,
                                                        sigma)
print 'DOCSP problem, reconstruction error = %.2f, time taken = %.2f' % \
    (reconstruction_error_DOCSP, time_taken)
plt.title('DOCSP, reconstruction error = %.2f' % reconstruction_error_DOCSP)
plt.plot(d)
plt.plot(-np.dot(A, solution_dual))
plt.plot(cost_vector_dual)
plt.legend(('d', 'sum_u', 'cost vector'), 'upper left')
plt.savefig('DOCSP_T=%d_N=%d_sigma=%.2f_noise=%.2f.png' % (T, N, sigma, NOISE_LEVEL_DOCSP),
            dpi = 300)
plt.close()

clock_measurements['DOCSP_low_noise'] = time_taken
error_measurements['DOCSP_low_noise'] = reconstruction_error_DOCSP

#
#    Solve with DOCSP (high noise in com)
#
NOISE_LEVEL_DOCSP = 0.5
start = time.clock()
solution_dual, obj_value_dual, cost_vector_dual, obj_values_dual = \
        dual_solver.solve(lower_matrix_list, 
                          upper_matrix_list, 
                          core_matrix_list, 
                          o_vect,
                          N, T, sigma, A,
                          noise_level = NOISE_LEVEL_DOCSP)
stop = time.clock()
time_taken = stop - start
plt.title('Control convergence DOCSP')
plt.plot(obj_values_dual)
plt.xlabel('Iteration')
plt.ylabel('Obj value')
plt.savefig('DOCSP_conv_T=%d_N=%d_sigma=%.2f_noise=%.2f.png' % (T, N, sigma, NOISE_LEVEL_DOCSP),
            dpi = 300)
plt.close()
reconstruction_error_DOCSP = utils.compute_original_obj(A, 
                                                        solution_dual,
                                                        o_vect,
                                                        sigma)
print 'DOCSP problem, reconstruction error = %.2f, time taken = %.2f' % \
        (reconstruction_error_DOCSP, time_taken)
plt.title('DOCSP, reconstruction error = %.2f' % reconstruction_error_DOCSP)
plt.plot(d)
plt.plot(-np.dot(A, solution_dual))
plt.plot(cost_vector_dual)
plt.legend(('d', 'sum_u', 'cost vector'), 'upper left')
plt.savefig('DOCSP_T=%d_N=%d_sigma=%.2f_noise=%.2f.png' % (T, N, sigma, NOISE_LEVEL_DOCSP),
            dpi = 300)
plt.close()

clock_measurements['DOCSP_high_noise'] = time_taken
error_measurements['DOCSP_high_noise'] = reconstruction_error_DOCSP

pickle.dump(clock_measurements, open('clock_meas_sigma_%.2f.pi' % sigma, 'wb'))
pickle.dump(error_measurements, open('error_meas_sigma_%.2f.pi' % sigma, 'wb'))


