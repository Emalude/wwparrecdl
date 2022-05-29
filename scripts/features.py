import sys
parameter = sys.argv[1]
coherence = sys.argv[2]

from pathlib import Path
import glob
import time
import numpy as np


parent_path = Path(__file__).resolve().parents[1].absolute()
data_dir = parent_path / 'wwparrecdl/data' / parameter / coherence
features_dir = parent_path / 'wwparrecdl/data/dl_input'

features = []
parameters = []


choices_files = data_dir / "choices/choices.npy"
times_files = data_dir / "times/times.npy"
parameters_files = data_dir / "parameters/parameters.npy"

start = time.time()


times = np.load(str(times_files))
choices = np.load(str(choices_files))
params = np.load(str(parameters_files))
params[:,2] /= 500

for i in range(len(times)):

    feature_point = []
    for p in range(choices.shape[1]):
        correct = choices[i][p] == 1
        incorrect = choices[i][p] == 0
        non_decision = choices[i][p] == -1
        decided = np.sum(correct) + np.sum(incorrect)

        if decided == 0:
            acc = -1
        else:
            acc = np.sum(correct)/decided

        indecis  = np.sum(non_decision)/np.shape(times[0][0])
        avg_rt_l = np.nan_to_num(np.mean(times[i][p][correct]/3000), nan = -1)
        avg_rt_r = np.nan_to_num(np.mean(times[i][p][incorrect]/3000), nan = -1)

        if indecis < 0.95:
            try:
                perc_left_10 = np.percentile(times[i][p][correct], 10)/3000
                perc_left_30 = np.percentile(times[i][p][correct], 30)/3000
                perc_left_50 = np.percentile(times[i][p][correct], 50)/3000
                perc_left_70 = np.percentile(times[i][p][correct], 70)/3000
                perc_left_90 = np.percentile(times[i][p][correct], 90)/3000
            except IndexError:
                perc_left_10 = perc_left_30 = perc_left_50 = perc_left_70 = perc_left_90 = -1

            try:
                perc_right_10 = np.percentile(times[i][p][incorrect], 10)/3000
                perc_right_30 = np.percentile(times[i][p][incorrect], 30)/3000
                perc_right_50 = np.percentile(times[i][p][incorrect], 50)/3000
                perc_right_70 = np.percentile(times[i][p][incorrect], 70)/3000
                perc_right_90 = np.percentile(times[i][p][incorrect], 90)/3000
            except IndexError:
                perc_right_10 = perc_right_30 = perc_right_50 = perc_right_70 = perc_right_90 = -1
        else:
            perc_left_10 = perc_left_30 = perc_left_50 = perc_left_70 = perc_left_90 = -1
            perc_right_10 = perc_right_30 = perc_right_50 = perc_right_70 = perc_right_90 = -1
        
        feature_point.append(np.array([acc, indecis, avg_rt_l, avg_rt_r,
                                       perc_left_10, perc_left_30, perc_left_50, perc_left_70, perc_left_90,
                                       perc_right_10, perc_right_30, perc_right_50, perc_right_70, perc_right_90
                                       ], dtype=object))
    features.append(feature_point)

end = time.time()

print(f'Generated features in {(end-start)/60} min. Saving results...')

np.save(str(features_dir / ('features_' + parameter + f'_{coherence}')), features)
np.save(str(features_dir / ('parameters_' + parameter + f'_{coherence}')), params)
