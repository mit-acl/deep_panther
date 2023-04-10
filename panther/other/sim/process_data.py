#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  *
#  * Read the bag files and process the data
#  * The data to extract is:
#  *    (1) travel time - the time it took to reach the goal
#  *    (2) computation time - the time it took to compute the trajectory
#  *    (3) number of collisions
#  *    (4) fov rate - the percentage of time the drone keeps obstacles in its FOV
#  *    (5) continuous fov detection (min, avg, max) - the minimum, average, and maximum of coninuous detection the drone keeps obstacles in its FOV
#  *    (6) translational dynamic constrate violation rate - the percentage of time the drone violates the translational dynamic constraints
#  *    (7) yaw dynamic constrate violation rate - the percentage of time the drone violates the yaw dynamic constraints
#  *    (8) success rate - the percentage of time the drone reaches the goal without any collision or dynamic constraint violation
#  *    (9) accel trajectory smoothness
#  *    (10) jerk trajectory smoothness
#  *    (11) number of stops
#  *    (12) number of turns
#  * -------------------------------------------------------------------------- */

import os
import sys
import rosbag
import rospy
import numpy as np
from statistics import mean
from tf_bag import BagTfTransformer

if __name__ == '__main__':

    ##
    ## Paramters
    ##

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/media/kota/T7/deep-panther/bags"
    TOPICS_TO_UNPACK = "/{}/goal /{}/state /tf /tf_static /{}/panther/fov /obstacles_mesh /{}/panther/best_solution_expert /{}/panther/best_solution_student /{}/term_goal /{}/panther/actual_traj /clock /trajs /sim_all_agents_goal_reached /{}/panther/is_ready /{}/panther/log"
    
    ##
    ## Loop over each simulation folder (e.g. /3_obs/)
    ##

    sim_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    sim_folders.sort()

    for sim_folder in sim_folders:
        
        ##
        ## Data extraction preparation
        ##

        # (1) travel time
        sim_start_times = []
        sim_end_times = []
        
        # (2) computation time
        computatoin_times = []

        # (3) number of collisions
        num_of_collisions = 0

        ##
        ## Read the bag files
        ##

        bag_files = [f for f in os.listdir(os.path.join(DATA_DIR, sim_folder)) if f.endswith(".bag")]
        bag_files.sort()

        for bag_file in bag_files:

            print(bag_file)
            agent_name = bag_file[8:13]
            topics = TOPICS_TO_UNPACK.format(*[agent_name for i in range(9)]).split(" ")

            with rosbag.Bag(os.path.join(DATA_DIR, sim_folder, bag_file)) as bag:

                ##
                ## (1) travel time
                ##

                for topic, msg, t in bag.read_messages(topics=topics):
                    if  topic == f"/{agent_name}/term_goal":
                        sim_start_times.append(msg.header.stamp.to_sec())
                    if topic == f"/sim_all_agents_goal_reached":
                        sim_end_times.append(msg.header.stamp.to_sec())
                ##
                ## (2) computation time
                ##

                for topic, msg, t in bag.read_messages(topics=topics):
                    if topic == f"/{agent_name}/panther/log":
                        computatoin_times.append(msg.ms_opt)

            
                # (3) number of collisions (using tf)

                bag_transformer = BagTfTransformer(bag)
                sim_start_time = sim_start_times[-1]
                sim_end_time = sim_end_times[-1]
                discrete_times = np.linspace(sim_start_time, sim_end_time, (sim_end_time - sim_start_time) * 100)






        ##
        ## Data extraction
        ##

        # (1) travel time
        travel_time = max(sim_end_times) - min(sim_start_times)

        # (2) computation time
        computation_time = mean(computatoin_times) # right now I am not including octopus search time

        # (3) number of collisions
        num_of_collisions = 0

        # # (4) fov rate
        # fov_rate = 0

        # # (5) continuous fov detection (min, avg, max)
        # continuous_fov_detection_min = 0
        # continuous_fov_detection_avg = 0
        # continuous_fov_detection_max = 0

        # # (6) translational dynamic constrate violation rate
        # translational_dynamic_constrate_violation_rate = 0

        # # (7) yaw dynamic constrate violation rate
        # yaw_dynamic_constrate_violation_rate = 0

        # # (8) success rate
        # success_rate = 0

        # # (9) accel trajectory smoothness
        # accel_trajectory_smoothness = 0

        # # (10) jerk trajectory smoothness
        # jerk_trajectory_smoothness = 0

        # # (11) number of stops
        # num_of_stops = 0

        # # (12) number of turns
        # num_of_turns = 0

        # ##
        # ## Save the data
        # ##

        # # (1) travel time
        # with open(os.path.join(DATA_DIR, sim_folder, "travel_time.txt"), "a") as f:
        #     f.write(str(travel_time) + "