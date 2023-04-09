
#!/usr/bin/env python

import subprocess
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import math, time, random
from scipy.spatial import Delaunay
from fov_detector import check_obst_is_in_FOV, rotate_vector, quaternion_multiply

if __name__ == "__main__":

    ##
    ## Run simluations
    ##

    # home directory
    HOME_DIR = sys.argv[1] if len(sys.argv) > 1 else "/media/kota/T7/deep-panther"

    # run simulations
    subprocess.run(["python", "run_many_sims.py", str(HOME_DIR)])
    
    # bags directory
    DATA_DIR = HOME_DIR + "/bags"
    
    # process data
    subprocess.run(["python", "process_data.py", str(DATA_DIR)])
    
    



