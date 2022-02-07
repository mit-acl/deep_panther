import numpy as np	
b = np.load('./evals/tmp_dagger/2/demos/round-002/dagger-demo-20220119_182704_bb5ed3.npz')
print("b['obs']=",b['obs']) #Note that this has all the observations until the termination of the environment + one extra observation
print("b['acts']=",b['acts'])
