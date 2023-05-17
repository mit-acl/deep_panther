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
import pickle
from pylab import *

##
##############################################################################
##

def plot_bar(y_axis_label, fig_name, data1, data2):

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
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()

##
##############################################################################
##

def organize_for_plot_box_1_traj_6_traj(lists):

    data1 = [[lists[0], lists[1]], [lists[2], lists[3]], lists[4]]
    data2 = [[lists[5], lists[6]], [lists[7], lists[8]], lists[9]]

    return data1, data2

##
##############################################################################
##

def sort_for_plot_box_1_traj_6_traj(data):

    list = []

    for folder in data:
        if "/1_traj/parm/" in folder:
            list.append(folder)
    
    for folder in data:
        if "/6_traj/parm/" in folder:
            list.append(folder)

    for folder in data:
        if "/1_traj/parm_star/" in folder:
            list.append(folder)

    for folder in data:
        if "/6_traj/parm_star/" in folder:
            list.append(folder)
    
    for folder in data:
        if "/6_traj/primer/" in folder:
            list.append(folder)
    
    return list
        
##
##############################################################################
##

# function for setting the colors of the box plots pairs (https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots)
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    # setp(bp['fliers'][0], color='blue')
    # setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    # setp(bp['fliers'][2], color='red')
    # setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

##
##############################################################################
##

# function for setting the colors of the box plots pairs (https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots)
def setBoxColors_for_primer(bp):
    setp(bp['boxes'][0], color='red')
    setp(bp['caps'][0], color='red')
    setp(bp['caps'][1], color='red')
    setp(bp['whiskers'][0], color='red')
    setp(bp['whiskers'][1], color='red')
    # setp(bp['fliers'][0], color='red')
    # setp(bp['fliers'][1], color='red')
    setp(bp['medians'][0], color='red')

##
##############################################################################
##

def plot_box_1_traj_6_traj(y_axis_label, fig_name, data1, data2):

    """ 
    plot the data in a box plot 
    
    data1: 1 agent with 2 obstacles
        ex. [[[1-traj-parm], [6-traj-parm]], [[1-traj-parm-star, 6-traj-parm-star]], [6-traj-primer]]
    data2: 3 agents with 2 obstacles
        ex. [[[1-traj-parm], [6-traj-parm]], [[1-traj-parm-star, 6-traj-parm-star]], [6-traj-primer]]
    """

    # constants
    LABELS = ["PARM", "PARM*", "PRIMER"]

    # get rid of the top and right frame
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    # plot data1
    fig = plt.figure(figsize =(10, 7))
    # hold(True)
 
    # Creating axes instance
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    # Creating plot
    # parm
    bp_parm_1 = ax1.boxplot(data1[0], positions=[1,1.5], widths=0.3, showfliers=False)
    bp_parm_2 = ax2.boxplot(data2[0], positions=[1,1.5], widths=0.3, showfliers=False)
    # parm*
    bp_parm_star_1 = ax1.boxplot(data1[1], positions=[2.5,3], widths=0.3, showfliers=False)
    bp_parm_star_2 = ax2.boxplot(data2[1], positions=[2.5,3], widths=0.3, showfliers=False)
    # primer
    bp_primer_1 = ax1.boxplot(data1[2], positions=[4], widths=0.3, showfliers=False)
    bp_primer_2 = ax2.boxplot(data2[2], positions=[4], widths=0.3, showfliers=False)
    # set colors
    setBoxColors(bp_parm_1)
    setBoxColors(bp_parm_2)
    setBoxColors(bp_parm_star_1)
    setBoxColors(bp_parm_star_2)
    setBoxColors_for_primer(bp_primer_1)
    setBoxColors_for_primer(bp_primer_2)
    # vertical line
    ax1.axvline(x=2, color='black', linestyle='--')
    ax1.axvline(x=3.5, color='black', linestyle='--')
    ax2.axvline(x=2, color='black', linestyle='--')
    ax2.axvline(x=3.5, color='black', linestyle='--')
    # make it log scale
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # add space between subplots
    plt.subplots_adjust(hspace=0.35)

    # make the plot prettier
    ax1.set_xticks([1.25, 2.75, 4])
    ax1.set_xticklabels(LABELS, fontproperties=font)
    ax2.set_xticks([1.25, 2.75, 4])
    ax2.set_xticklabels(LABELS, fontproperties=font)
    ax1.set_xlim(0.5, 4.5)
    ax2.set_xlim(0.5, 4.5)
    fig.text(0.06, 0.5, y_axis_label, va='center', rotation='vertical', fontproperties=font)
    ax1.set_title("1 agent with 2 obstacles", fontproperties=font)
    ax2.set_title("3 agents with 2 obstacles", fontproperties=font)
    
    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1,1],'b-')
    hR, = plot([1,1],'r-')
    ax1.legend((hB, hR),('1 traj', '6 trajs'))
    ax2.legend((hB, hR),('1 traj', '6 trajs'))
    hB.set_visible(False)
    hR.set_visible(False)

    # save figure
    plt.savefig(fig_name, bbox_inches='tight')

def plot_box(y_axis_label, fig_name, data1, data2):

    # constants
    LABELS = ["PARM", "PARM*", "PRIMER"]
    X_AXIS_STEP = 1
    x_axis = np.arange(1, X_AXIS_STEP*len(LABELS)+1, X_AXIS_STEP) 

    # get rid of the top and right frame
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    # plot data1
    fig = plt.figure(figsize =(10, 7))
 
    # Creating axes instance
    ax1 = fig.add_subplot(2,1,1)
    # ax1.grid(axis='y')
    ax2 = fig.add_subplot(2,1,2)
    # ax2.grid(axis='y')

    # Creating plot
    bp1 = ax1.boxplot(data1, showfliers=False)
    bp2 = ax2.boxplot(data2, showfliers=False)

    # add space between subplots
    plt.subplots_adjust(hspace=0.35)

    # make the plot prettier
    ax1.set_xticklabels(LABELS, fontproperties=font)
    ax2.set_xticklabels(LABELS, fontproperties=font)
    fig.text(0.06, 0.5, y_axis_label, va='center', rotation='vertical', fontproperties=font)
    ax1.set_title("1 agent with 2 obstacles", fontproperties=font)
    ax2.set_title("2 agents with 2 obstacles", fontproperties=font)
    
    # save figure
    plt.savefig(fig_name, bbox_inches='tight')

# def plot_box(y_axis_label, fig_name, data1, data2):

#     # constants
#     LABELS = ["Expert", "Student"]
#     X_AXIS_STEP = 1
#     x_axis = np.arange(1, X_AXIS_STEP*len(LABELS)+1, X_AXIS_STEP) 

#     # get rid of the top and right frame
#     mpl.rcParams['axes.spines.right'] = False
#     mpl.rcParams['axes.spines.top'] = False

#     # plot data1
#     fig = plt.figure(figsize =(10, 7))
 
#     # Creating axes instance
#     ax1 = fig.add_subplot(2,1,1)
#     # ax1.grid(axis='y')
#     ax2 = fig.add_subplot(2,1,2)
#     # ax2.grid(axis='y')

#     # Creating plot
#     bp1 = ax1.boxplot(data1, showfliers=False)
#     bp2 = ax2.boxplot(data2, showfliers=False)

#     # add space between subplots
#     plt.subplots_adjust(hspace=0.35)

#     # make the plot prettier
#     ax1.set_xticklabels(LABELS, fontproperties=font)
#     ax2.set_xticklabels(LABELS, fontproperties=font)
#     fig.text(0.06, 0.5, y_axis_label, va='center', rotation='vertical', fontproperties=font)
#     ax1.set_title("1 agent with 2 obstacles", fontproperties=font)
#     ax2.set_title("2 agents with 2 obstacles", fontproperties=font)
    
#     # save figure
#     plt.savefig(fig_name, bbox_inches='tight')

##
##############################################################################
##

if __name__ == '__main__':

    ##
    ## Paramters
    ##

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/media/kota/T7/deep-panther/bags"
    FIG_SAVE_DIR = DATA_DIR + "/figs"

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

    sim_folders = sort_for_plot_box_1_traj_6_traj(two_obs_one_agent_list)
    sim_folders.extend(sort_for_plot_box_1_traj_6_traj(two_obs_three_agents_list))

    for sim_folder in sim_folders:

        # (1) travel time
        with open(os.path.join(sim_folder, "pkls", "travel_time_list.pkl"), "rb") as f:
            travel_time_list.append(pickle.load(f))
        
        # (2) computation time
        with open(os.path.join(sim_folder, "pkls", "computation_time_list.pkl"), "rb") as f:
            computation_time_list.append(pickle.load(f))
        
        # (3) number of collisions
        with open(os.path.join(sim_folder, "pkls", "num_of_collisions_btwn_agents_list.pkl"), "rb") as f:
            num_of_collisions_btwn_agents_list.append(pickle.load(f))
        
        with open(os.path.join(sim_folder, "pkls", "num_of_collisions_btwn_agents_and_obstacles_list.pkl"), "rb") as f:
            num_of_collisions_btwn_agents_and_obstacles_list.append(pickle.load(f))
        
        # (4) fov rate
        with open(os.path.join(sim_folder, "pkls", "fov_rate_list.pkl"), "rb") as f:
            fov_rate_list.append(pickle.load(f))
        
        # (5) continuous fov detection
        with open(os.path.join(sim_folder, "pkls", "continuous_fov_detection_list.pkl"), "rb") as f:
            continuous_fov_detection_list.append(pickle.load(f))
        
        # (6) translational dynamic constraint violation rate
        with open(os.path.join(sim_folder, "pkls", "translational_dynamic_constraint_violation_rate_list.pkl"), "rb") as f:
            translational_dynamic_constraint_violation_rate_list.append(pickle.load(f))
        
        # (7) yaw dynamic constraint violation rate
        with open(os.path.join(sim_folder, "pkls", "yaw_dynamic_constraint_violation_rate_list.pkl"), "rb") as f:
            yaw_dynamic_constraint_violation_rate_list.append(pickle.load(f))
        
        # (8) success rate
        with open(os.path.join(sim_folder, "pkls", "success_rate_list.pkl"), "rb") as f:
            success_rate_list.append(pickle.load(f))
        
        # (9) accel trajectory smoothness
        with open(os.path.join(sim_folder, "pkls", "accel_trajectory_smoothness_list.pkl"), "rb") as f:
            accel_trajectory_smoothness_list.append(pickle.load(f))
        
        # (10) jerk trajectory smoothness
        with open(os.path.join(sim_folder, "pkls", "jerk_trajectory_smoothness_list.pkl"), "rb") as f:
            jerk_trajectory_smoothness_list.append(pickle.load(f))
        
        # (11) number of stops
        with open(os.path.join(sim_folder, "pkls", "num_of_stops_list.pkl"), "rb") as f:
            num_of_stops_list.append(pickle.load(f))
        
        print("Finished extracting data from {}".format(sim_folder))

    ##
    ## Plot data (box plot)
    ##

    # if there is not file to save figures, create one
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)

    # (1) travel time
    travel_time_y_axis_label = "Travel time [s]"
    travel_time_fig_name = "travel_time.pdf"
    data1, data2 = organize_for_plot_box_1_traj_6_traj(travel_time_list)
    plot_box_1_traj_6_traj(y_axis_label=travel_time_y_axis_label, fig_name=FIG_SAVE_DIR+"/"+travel_time_fig_name, data1=data1, data2=data2)

    # (2) computation time
    computation_time_y_axis_label = "Computation time [s]"
    computation_time_fig_name = "computation_time.pdf"
    data1, data2 = organize_for_plot_box_1_traj_6_traj(computation_time_list)
    plot_box_1_traj_6_traj(y_axis_label=computation_time_y_axis_label, fig_name=FIG_SAVE_DIR+"/"+computation_time_fig_name, data1=data1, data2=data2)

    # (3) number of collisions
    # num_of_collisions_btwn_agents_y_axis_label = "Number of collisions between agents"
    # num_of_collisions_btwn_agents_fig_name = "num_of_collisions_btwn_agents"
    # num_of_collisions_btwn_agents_list.append(num_of_collisions_btwn_agents_list[0])
    # num_of_collisions_btwn_agents_list.append(num_of_collisions_btwn_agents_list[0])
    # plot_box(y_axis_label=num_of_collisions_btwn_agents_y_axis_label, fig_name=FIG_SAVE_DIR+"/"+num_of_collisions_btwn_agents_fig_name, data1=num_of_collisions_btwn_agents_list, data2=num_of_collisions_btwn_agents_list)

    # num_of_collisions_btwn_agents_and_obstacles_y_axis_label = "Number of collisions between agents and obstacles"
    # num_of_collisions_btwn_agents_and_obstacles_fig_name = "num_of_collisions_btwn_agents_and_obstacles"
    # num_of_collisions_btwn_agents_and_obstacles_list.append(num_of_collisions_btwn_agents_and_obstacles_list[0])
    # num_of_collisions_btwn_agents_and_obstacles_list.append(num_of_collisions_btwn_agents_and_obstacles_list[0])
    # plot_box(y_axis_label=num_of_collisions_btwn_agents_and_obstacles_y_axis_label, fig_name=FIG_SAVE_DIR+"/"+num_of_collisions_btwn_agents_and_obstacles_fig_name, data1=num_of_collisions_btwn_agents_and_obstacles_list, data2=num_of_collisions_btwn_agents_and_obstacles_list)

    # (4) fov rate (this is 1 or 0 and not good for box plot)
    # fov_rate_y_axis_label = "FOV rate [/%]"
    # fov_rate_list_fig_name = "fov_rate"
    # fov_rate_list = [[int(elem) for elem in elem_list] for elem_list in fov_rate_list]
    # print(fov_rate_list)
    # fov_rate_list.append(fov_rate_list[0])
    # fov_rate_list.append(fov_rate_list[0])
    # plot_box(y_axis_label=fov_rate_y_axis_label, fig_name=FIG_SAVE_DIR+"/"+fov_rate_list_fig_name, data1=fov_rate_list, data2=fov_rate_list)

    # (5) continuous fov detection (we are only recording the max fov detection, so no need to plot )
    # continuous_fov_detection_y_axis_label = "Continuous FOV detection [%]"
    # continuous_fov_detection_fig_name = "continuous_fov_detection"
    # continuous_fov_detection_list.append(continuous_fov_detection_list[0])
    # continuous_fov_detection_list.append(continuous_fov_detection_list[0])
    # plot_box(y_axis_label=continuous_fov_detection_y_axis_label, fig_name=FIG_SAVE_DIR+"/"+continuous_fov_detection_fig_name, data1=continuous_fov_detection_list, data2=continuous_fov_detection_list)

    # (9) accel trajectory smoothness
    accel_trajectory_smoothness_y_axis_label = r"$\int\left\Vert\mathbf{a}\right\Vert^2dt \ [m^2/s^3]$" #"Accel trajectory smoothness [m/s^3]"
    accel_trajectory_smoothness_fig_name = "accel_trajectory_smoothness.pdf" 
    data1, data2 = organize_for_plot_box_1_traj_6_traj(accel_trajectory_smoothness_list)
    plot_box_1_traj_6_traj(y_axis_label=accel_trajectory_smoothness_y_axis_label, fig_name=FIG_SAVE_DIR+"/"+accel_trajectory_smoothness_fig_name, data1=data1, data2=data2)

    # (10) jerk trajectory smoothness
    jerk_trajectory_smoothness_y_axis_label = r"$\int\left\Vert\mathbf{j}\right\Vert^2dt \ [m^2/3^5]$" #"Jerk trajectory smoothness"
    jerk_trajectory_smoothness_fig_name = "jerk_trajectory_smoothness.pdf"
    data1, data2 = organize_for_plot_box_1_traj_6_traj(jerk_trajectory_smoothness_list)
    plot_box_1_traj_6_traj(y_axis_label=jerk_trajectory_smoothness_y_axis_label, fig_name=FIG_SAVE_DIR+"/"+jerk_trajectory_smoothness_fig_name, data1=data1, data2=data2)

    # # (11) number of stops
    # num_of_stops_y_axis_label = "Number of stops"
    # num_of_stops_fig_name = "num_of_stops"
    # plot_box(y_axis_label=num_of_stops_y_axis_label, fig_name=FIG_SAVE_DIR+"/"+num_of_stops_fig_name, data1=num_of_stops_list[0:2], data2=num_of_stops_list[2:4])
