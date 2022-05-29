from __future__ import absolute_import, division, print_function

DEF num_samples = 5000
DEF num_trials = 10000
DEF num_points = 1 #number of repetition of the same sample (parameter combination).
                   #See README for possible use cases.

import cython
import numpy as np
cimport numpy as np

from libc.math cimport sqrt, exp, log
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

np.import_array()

cdef float a           = 270.0
cdef float b           = 108.0
cdef float d           = 0.154
cdef float gamma       = 0.641 / 1000
cdef float tau_s       = 100.0 #ms
cdef float tau_noise   = 2.0   #ms
#cdef float Jii         = 0.2601  # 0.1561
#cdef float Jij         = 0.0497  # 0.0264
cdef float J_ext       = 0.00052 # 0.0002243
cdef float beta        = 1
#cdef int ndt           = 0
cdef float I_o         = 0.329
cdef float sigma_noise = I_o / 16.275
cdef int mu_o          = 15
cdef int dt            = 1
#cdef int threshold     = 15
cdef int coherence     = 15

srand(time(NULL))

@cython.cdivision(True)
def wangwong():

    """
    Generates samples of the Wong & Wang (2006) model.
    Each sample uses a different combination of parameters and consists of 10,000 trials of 2500ms each.
    """

    # Allocate arrays and variables
    cdef np.ndarray[np.int_t, ndim=3] choices = np.empty((num_samples, 4, num_trials), dtype = int)
    cdef np.ndarray[np.int_t, ndim=3] times = np.empty((num_samples, 4, num_trials), dtype = int)
    cdef np.ndarray[np.float_t, ndim=2] parameters = np.empty((num_samples, 3), dtype = float)
    
    cdef int s, j, p
    cdef int t

    cdef float x1, x2, w

    cdef float I_l, I_r
    cdef float r_l, r_r
    cdef float sl_hz, sr_hz

    cdef float sl, sr
    cdef float I_n1, I_n2

    cdef float I_mot_l, I_mot_r
    
    cdef float Jii
    cdef float Jij
    #cdef float beta
    cdef int ndt
    cdef float threshold

    for s in range(num_samples):
        
        # Randomly draw parameters from uniform distributions.
        # if x is a random number between 0 and 1, to get a random number between min and max:
        # x*(max - min)/min
        Jii   = <float>rand()/RAND_MAX * (0.30 - 0.25) + 0.25
        Jij   = <float>rand()/RAND_MAX *  0.25
        ndt = rand() % 400 + 100
        threshold = <float>rand()/RAND_MAX * (20 - 15) + 15
        #beta = <float>rand()/RAND_MAX * (1.05 - 0.95) + 0.95
        
        parameters[s,0] = Jii
        parameters[s,1] = Jij
        parameters[s,2] = ndt
        parameters[s,3] = threshold

        for c in range(num_points):
            for j in range(num_trials):
                I_mot_l = J_ext * mu_o * (1.0 + coherence * 1.0 / 100.0)
                I_mot_r = J_ext * mu_o * (1.0 - coherence * 1.0 / 100.0)

                sl   = <double>rand()/RAND_MAX * 0.1
                sr   = <double>rand()/RAND_MAX * 0.1
                I_n1 = <double>rand()/RAND_MAX * 0.1
                I_n2 = <double>rand()/RAND_MAX * 0.1

                times[s,c,j] = 2500
                choices[s,c,j] = -1

                for t in range(2500):
                 
                    I_l = Jii * sl - Jij*sr + I_mot_l + beta*I_o + I_n1
                    I_r = Jii * sr - Jij*sl + I_mot_r + beta*I_o + I_n2

                    r_l = (a*I_l - b)/(1.0 - exp(-d*(a*I_l - b)))
                    r_r = (a*I_r - b)/(1.0 - exp(-d*(a*I_r - b)))

                    sl = sl + (-sl/ tau_s + (1.0 - sl) * gamma * r_l)*dt
                    sr = sr + (-sr/ tau_s + (1.0 - sr) * gamma * r_r)*dt
                    
                    #Box-Muller transformation for random gaussian number generations
                    #The numbers are used for noisy currents calculation below
                    w = 2.0
                    while (w >= 1.0):
                        x1 = 2.0 * rand()/RAND_MAX - 1.0
                        x2 = 2.0 * rand()/RAND_MAX - 1.0
                        w = x1 * x1 + x2 * x2
                    w = sqrt((-2.0 * log(w)) / w)

                    I_n1 = I_n1 + (- I_n1 + x1*w * sqrt(tau_noise) * sigma_noise) * dt / tau_noise
                    I_n2 = I_n2 + (- I_n2 + x2*w * sqrt(tau_noise) * sigma_noise) * dt / tau_noise

                    sl_hz = sl / ((1.0 - sl) * gamma * tau_s)
                    sr_hz = sr / ((1.0 - sr) * gamma * tau_s)

                    if threshold - sl_hz < 1e-9 or threshold - sr_hz < 1e-9:
                        times[s,c,j] = t + ndt
                        choices[s,c,j] = sl_hz > sr_hz
                        break

    return times, choices, parameters
