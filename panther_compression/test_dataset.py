# test file for traing/validation/test dataset
# this file tests the dataset class

#  import
import torch as th
import numpy as np
import os

# path
path = '/home/kota/Research/deep-panther_ws/src/deep_panther/panther_compression/evals/tmp_dagger/2/demos/round-000'

# get a list of npz files and load data
npz_files = np.array([f for f in os.listdir(path) if f.endswith('.npz')])

# load data that is in npz files
data = [np.load(os.path.join(path, f)) for f in npz_files]

# ['obs', 'acts', 'infos', 'terminal', 'rews']
for i in range(1,10):
    print(data[i]['obs'].shape)
    print(data[i]['acts'].shape)
# # define Dataset class