import numpy as np	
b = np.load('./evals/tmp_dagger/2/demos/round-000/dagger-demo-20220118_192929_1cfa82.npz')
print("b['obs']=",b['obs']) #Note that this has two elements (previous obs and next obs)
print("b['acts']=",b['acts'])
