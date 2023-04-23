#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  *
#  * Read the text file and plot the data
#  * The data to plot is:
#  *    (1) travel time - the time it took to reach the goal
#  *    (2) computation time - the time it took to compute the trajectory
#  *    (3) number of collisions
#  *    (4) fov rate - the percentage of time the drone keeps obstacles in its FOV when the drone is actually close to the obstacles
#  *    (5) continuous fov detection - the minimum, average, and maximum of coninuous detection the drone keeps obstacles in its FOV
#  *    (6) translational dynamic constraint violation rate - the percentage of time the drone violates the translational dynamic constraints
#  *    (7) yaw dynamic constraint violation rate - the percentage of time the drone violates the yaw dynamic constraints
#  *    (8) success rate - the percentage of time the drone reaches the goal without any collision or dynamic constraint violation
#  *    (9) accel trajectory smoothness
#  *    (10) jerk trajectory smoothness
#  *    (11) number of stops
#  * -------------------------------------------------------------------------- */

import os
import sys
import rosbag
import rospy
import numpy as np
from statistics import mean
import tf_bag
from tf_bag import BagTfTransformer
import itertools
from fov_detector import check_obst_is_in_FOV
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from process_data import listdirs

def plot_bar(y_axis_label, data1, data2):

    # constants
    LABELS = ["PARM", "PARM*", "PRIMER"]
    X_AXIS_STEP = 0.5
    x_axis = np.arange(0, X_AXIS_STEP*len(LABELS), X_AXIS_STEP) 

    # get rid of the top and right frame
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    # fig, (ax1, ax2) = plt.subplots(2,1)

    # plot data1
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(2,1,1)
    plt.subplot(2,1,1)
    bars = []
    bars.append(plt.bar(x = x_axis[:-1] - 0.1, height = data1[0], width=0.2, label="1 traj", color="tomato", edgecolor="black"))
    bars.append(plt.bar(x = x_axis[:-1] + 0.1, height = data1[1], width=0.2, label="6 trajs", color="lime", edgecolor="black"))
    bars.append(plt.bar(x = x_axis[-1], height = data1[2], width=0.2, color="lime", edgecolor="black"))

    bar1 = bars[0]
    bar2 = bars[1]
    plt.legend([bar1, bar2], ["1 traj", "6 trajs"], prop=font)
    # add numbers on top of bars
    for bar in bars:
        ax.bar_label(bar, padding=3, fontproperties=font)
    plt.xticks(x_axis, LABELS, fontproperties=font)
    plt.ylabel(y_axis_label, fontproperties=font)
    plt.title("1 agent with 2 obstacles", fontproperties=font)

    # add space between subplots
    plt.subplots_adjust(hspace=0.5)

    # plot data2
    ax = fig.add_subplot(2,1,2)
    plt.subplot(2,1,2)
    bars = []
    bars.append(plt.bar(x = x_axis[:-1] - 0.1, height = data2[0], width=0.2, label="1 traj", color="tomato", edgecolor="black"))
    bars.append(plt.bar(x = x_axis[:-1] + 0.1, height = data2[1], width=0.2, label="6 trajs", color="lime", edgecolor="black"))
    bars.append(plt.bar(x = x_axis[-1], height = data2[2], width=0.2, color="lime", edgecolor="black"))

    # add numbers on top of bars
    for bar in bars:
        ax.bar_label(bar, padding=3, fontproperties=font)

    # labels
    plt.xticks(x_axis, LABELS, fontproperties=font)
    plt.ylabel(y_axis_label, fontproperties=font)
    plt.title("3 agents with 2 obstacles", fontproperties=font)
    plt.show()

if __name__ == '__main__':

    ##
    ## Paramters
    ##

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/media/kota/T7/deep-panther/bags"

    ##
    ## Font setting
    ##

    font = font_manager.FontProperties()
    font.set_family('serif')
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams["font.family"] = "Times New Roman"
    font.set_size(16)

    ##
    ## Get simulation folders
    ##

    sim_folders = listdirs(rootdir=DATA_DIR, subdirs=[])

    ##
    ## Data extraction preparation for all simulation folders
    ##

    # (1) travel time
    travel_time_list = []

    # (2) computation time
    computation_time_list = []

    # (3) number of collisions
    num_of_collisions_btwn_agents_list = []
    num_of_collisions_btwn_agents_and_obstacles_list = []

    # (4) fov rate
    fov_rate_list = []

    # (5) continuous fov detection
    continuous_fov_detection_list = []

    # (6) translational dynamic constraint violation rate
    translational_dynamic_constraint_violation_rate_list = []

    # (7) yaw dynamic constraint violation rate
    yaw_dynamic_constraint_violation_rate_list = []

    # (8) success rate
    success_rate_list = []

    # (9) accel trajectory smoothness
    accel_trajectory_smoothness_list = []

    # (10) jerk trajectory smoothness
    jerk_trajectory_smoothness_list = []

    # (11) number of stops
    num_of_stops_list = []

    ##
    ## Extract data
    ##

    two_obs_one_agent_list = []
    two_obs_three_agents_list = [] 
    for sim_folder in sim_folders:
        if sim_folder.endswith("2_obs_1_agents"):
            two_obs_one_agent_list.append(sim_folder)
        elif sim_folder.endswith("2_obs_3_agents"):
            two_obs_three_agents_list.append(sim_folder)

    print("two_obs_one_agent_list: ", two_obs_one_agent_list)
    print("two_obs_three_agents_list: ", two_obs_three_agents_list)

    for sim_folders in [two_obs_one_agent_list, two_obs_three_agents_list]:

        for sim_folder in sim_folders:

            with open(os.path.join(sim_folder, "data.txt"), "r") as f:
                info = f.readlines()
                # (1) travel time
                travel_time_list.append(float(info[4].split(":")[1].strip().split(" ")[0]))

                # (2) computation time
                computation_time_list.append(float(info[5].split(":")[1].strip().split(" ")[0]))

                # (3) number of collisions
                num_of_collisions_btwn_agents_list.append(float(info[6].split(":")[1].strip()))
                num_of_collisions_btwn_agents_and_obstacles_list.append(float(info[7].split(":")[1].strip()))
        
                # (4) fov rate
                fov_rate_list.append(float(info[8].split(":")[1].strip().split(" ")[0]))

                # (5) continuous fov detection
                continuous_fov_detection_list.append(float(info[9].split(":")[1].strip()))

                # (6) translational dynamic constraint violation rate
                translational_dynamic_constraint_violation_rate_list.append(float(info[10].split(":")[1].strip().split(" ")[0]))

                # (7) yaw dynamic constraint violation rate
                yaw_dynamic_constraint_violation_rate_list.append(float(info[11].split(":")[1].strip().split(" ")[0]))

                # (8) success rate
                success_rate_list.append(float(info[12].split(":")[1].strip().split(" ")[0]))

                # (9) accel trajectory smoothness
                accel_trajectory_smoothness_list.append(float(info[13].split(":")[1].strip()))

                # (10) jerk trajectory smoothness
                jerk_trajectory_smoothness_list.append(float(info[14].split(":")[1].strip()))

                # (11) number of stops
                num_of_stops_list.append(float(info[15].split(":")[1].strip()))
            
            print("Finished extracting data from {}".format(sim_folder))

    ##
    ## change data format for plotting
    ## (eg. travel_time_list = [[parm's 1 traj, parm's 6 trajs], [parm star's 1 traj, parm star's 6 trajs], [primer]])
    ## travel_time_list_2obs1agents = [[1, 2], [3, 4], [5]]
    ## travel_time_list_2obs3agents = [[2, 3], [4, 5], [6]]
    ##

    ##
    ## Plot data
    ##

    # (1) travel time
    travel_time_list_2obs1agents = [travel_time_list[0:2], travel_time_list[2:4], travel_time_list[4]]
    travel_time_list_2obs3agents = [travel_time_list[5:7], travel_time_list[7:9], travel_time_list[9]]
    travel_time_y_axis_label = "Travel time [s]"
    plot_bar(y_axis_label=travel_time_y_axis_label, data1=travel_time_list_2obs1agents, data2=travel_time_list_2obs3agents)

    




    