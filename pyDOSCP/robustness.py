'''
Created on Dec 15, 2014

@author: Francois Belletti
'''

import cPickle as pickle
from matplotlib import pyplot as plt
import numpy as np

def compute_rel_precision(result_dict):
    original_error = result_dict['original']
    #
    primal_error = result_dict['primal']
    primal_prec = primal_error / original_error
    #
    DOCSP_error = result_dict['DOCSP']
    DOCSP_prec = DOCSP_error / original_error
    #
    DOCSP_low_noise_error = result_dict['DOCSP_low_noise']
    DOCSP_low_noise_prec = DOCSP_low_noise_error / original_error
    #
    DOCSP_high_noise_error = result_dict['DOCSP_high_noise']
    DOCSP_high_noise_prec = DOCSP_high_noise_error / original_error
    return [primal_prec, DOCSP_prec, DOCSP_low_noise_prec, DOCSP_high_noise_prec]

generic_file_name = 'error_meas_sigma_%.2f.pi'
prec_list_low_sigma = compute_rel_precision(pickle.load(open(generic_file_name % 0.1, 'rb')))
prec_list_medium_sigma = compute_rel_precision(pickle.load(open(generic_file_name % 0.5, 'rb')))
prec_list_high_sigma = compute_rel_precision(pickle.load(open(generic_file_name % 1.0, 'rb')))

plt.title('Relative error')
plt.plot(prec_list_low_sigma, c = 'black')
plt.plot(prec_list_medium_sigma, c = 'blue')
plt.plot(prec_list_high_sigma, c = 'orange')
plt.xticks(range(4), ['Primal',
                      'DOCSP',
                      'DOCSP noise = 0.1',
                      'DOCSP noise = 0.5'])
plt.legend(('Sigma = 0.1', 'Sigma = 0.5', 'Sigma = 1.0'), 'upper left')
plt.ylim((1.0, 1.10))
plt.savefig('Relative_errors.png', dpi = 300)
plt.close()


