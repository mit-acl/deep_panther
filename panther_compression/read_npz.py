#Jesus Tordesillas, December 2021

import numpy as np
import os
from imitation.algorithms import bc
from colorama import init, Fore, Back, Style
import re 
import matplotlib.pyplot as plt
import numpy as np

#See https://stackoverflow.com/a/2669120/6057617
def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
######################################

#See https://stackoverflow.com/a/25061573/6057617
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
######################################

path_seed='./evals/tmp_dagger/2/'
policies = sorted_nicely([f for f in os.listdir(path_seed) if os.path.isfile(os.path.join(path_seed, f))])
# print(policies)
# exit()

path_demos=path_seed+'demos/'
demos=sorted(os.listdir(path_demos))
# print(sorted(demos))
# exit()
round_num=0;
all_pairs=[];
all_errors=[];


x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)
plt.ion()# You probably won't need this if you're embedding things in a tkinter plot...
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma


for demo in demos:

    path_demo=path_demos + demo+'/'
    print("Adding files"+path_demo+Style.BRIGHT+Fore.BLUE+f"  [{len(all_pairs)} pairs in total]"+Style.RESET_ALL )
    pairs_files = os.listdir(path_demo)
    for pair_file in pairs_files:
        with np.load(path_demo + pair_file) as b:
            all_pairs.append({'obs':b['obs'][0], 'acts':b['acts'] })


    policy=policies[round_num];
    print("Using policy " + policies[round_num])
    student_policy = bc.reconstruct_policy(path_seed+policies[round_num]) #final_policy.pt

    round_num=round_num+1
    error=0;
    for pair in all_pairs:
        # print("pair= ",pair)
        # print("pair['obs']= ",pair['obs'])
        # with suppress_stdout():
        action_student = student_policy.predict(pair['obs'], deterministic=True)
        print("action_student[0].shape = ", action_student[0].shape)
        print("pair['acts'].shape = ", pair['acts'].shape)
        
        action_student=action_student[0].reshape(pair['acts'].shape)
        # action_student=action_student.reshape(1,-1)
        # print(action_student)
        # print(pair['acts'])
        norm_diff_pair=np.linalg.norm(action_student-pair['acts'])
        print("norm_diff_pair= ",norm_diff_pair)

        error = error + norm_diff_pair**2
    error=error/len(all_pairs)
    all_errors.append(error)
    line1.set_ydata(all_errors)
    line1.set_xdata(range(len(all_errors)))
    fig.canvas.draw()
    fig.canvas.flush_events()
    print("Error=", error)