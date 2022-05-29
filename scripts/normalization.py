'''import sys
PATH = '/home/c1978241/miniconda3/envs/flow/lib/python3.7/site-packages'
sys.path.insert(0, PATH)'''

import os
import numpy as np

current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
parameters_file = parent_path + '/data/dl_input/parameters.npy'

params = np.load(parameters_file)

scaled_Jii   = (params[:,0] - 0.25) / (0.30 - 0.25)
scaled_Jij   =  params[:,1] / 0.25
scaled_J_ext = (params[:,2] - 0.0001) / (0.0007 - 0.0001)
scaled_parameters = np.stack((scaled_Jii, scaled_Jij, scaled_J_ext)).T
        
print("done")

np.save(parent_path + '/data/dl_input/parameters_scaled.npy', scaled_parameters)


