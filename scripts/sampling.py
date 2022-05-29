'''import sys
PATH = '/home/c1978241/miniconda3/envs/flow/lib/python3.7/site-packages'
sys.path.insert(0, PATH)'''

import wwparrecdl as pr

import multiprocessing
from joblib import Parallel, delayed

import os
import time
import random
import numpy as np

#from scipy.io import savemat

current_path = os.getcwd()
data_dir = current_path + '/wwparrecdl/data/'

num_cores   = multiprocessing.cpu_count()
num_samples = 100
num_trials  = 10000
timestep    = 1 #ms
noise_mult  = 1
coherence   = 10
threshold   = 15 #Hz

def parallel_outcomes(func, num_trials, coherence):
    outcomes = []
    for j in range(num_samples):
        # a           = 270.
        # b           = 108.
        # d           = 0.154
        # gamma       = 0.641 / 1000
        # tau_s       = 100. #ms
        # tau_noise   = 2.   #ms
        Jll = Jrr     = random.choice(np.arange(0.1, 0.31, 0.02))
        Jlr = Jrl     = random.choice(np.arange(0.02, 0.051, 0.005))
        J_ext         = random.choice(np.arange(0.0002, 0.0005, 0.02))
        # I_o         = 0.329
        # sigma_noise = I_o / 16.275
        # mu_o        = np.random.choice(np.arange(0,71,10))
        # threshold   = 15
        # dt          = 1
        outcomes.append((Parallel(n_jobs=num_cores, backend="threading")(delayed(func)(coherence) for i in range(num_trials)), np.array([Jll, Jlr, J_ext])))
    return outcomes

start = time.time()
outcomes = parallel_outcomes(pr.wangwong, num_trials, coherence)
end = time.time()

print("Generated {} samples of {} trials each in {} min using {} cores.".format(num_samples, num_trials, (end-start)/60, num_cores))

#savemat(data_dir + "sample_{}_1.mat".format(coherence), {"outcomes":outcomes})
