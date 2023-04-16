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

if __name__ == '__main__':

    ##
    ## Paramters
    ##

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "bags"
    TOPICS_TO_UNPACK = "/{}/goal /{}/state /tf /tf_static /{}/panther/fov /obstacles_mesh /{}/panther/best_solution_expert /{}/panther/best_solution_student /{}/term_goal /{}/panther/actual_traj /clock /trajs /sim_all_agents_goal_reached /{}/panther/is_ready /{}/panther/log"
    NUM_OF_AGENTS = 2 #TODO: hard coded
    NUM_OF_OBSTACLES = 2 #TODO: hard coded
    AGENTS_LIST = [f"SQ{str(i+1).zfill(2)}s" for i in range(NUM_OF_AGENTS)]
    OBSTACLES_LIST = [f"obs_{4000+i}" for i in range(NUM_OF_OBSTACLES)]
    AGENT_BBOX = np.array([1.2, 1.2, 1.2]) #TODO:hard coded
    OBSTACLE_BBOX = np.array([0.6, 0.6, 0.3]) #TODO:hard coded    
    BBOX_AGENT_AGENT = AGENT_BBOX / 2  + AGENT_BBOX / 2
    BBOX_AGENT_OBST = AGENT_BBOX / 2 + OBSTACLE_BBOX / 2
    FOV_X_DEG = 76.0 #TODO:hard coded
    FOV_Y_DEG = 47.0 #TODO:hard coded
    FOV_DEPTH = 5.0 #TODO:hard coded
    MAX_VEL = 2.0 #TODO:hard coded
    MAX_ACC = 3.0 #TODO:hard coded
    MAX_JERK = 4.0 #TODO:hard coded
    MAX_DYAW = 1.0 #TODO:hard coded
    DT = 0.01 #TODO:hard coded

    ##
    ## Loop over each simulation folder (e.g. /3_obs/)
    ##

    sim_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    sim_folders.sort()

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
    ## Loop over each simulation folder (e.g. /3_obs/)
    ##

    num_of_bags = 0

    for sim_folder in sim_folders:
        
        ##
        ## Data extraction preparation for each simulation folder
        ##

        # (1) travel time
        
        # (2) computation time

        # (3) number of collisions
        num_of_collisions_btwn_agents = 0
        num_of_collisions_btwn_agents_and_obstacles = 0
        
        # (4) fov rate
        fov_rate = { agent: 0 for agent in AGENTS_LIST }
        agent_obstacle_proxy = { agent: 0 for agent in AGENTS_LIST }

        # (5) continuous fov detection (min, avg, max)
        continuous_fov_detection = { agent: { obstacle: 0 for obstacle in OBSTACLES_LIST } for agent in AGENTS_LIST }

        # (6) translational dynamic constraint violation rate
        translational_dynamic_constraint_violation_rate = 0

        # (7) yaw dynamic constraint violation rate
        yaw_dynamic_constraint_violation_rate = 0

        # (9) accel trajectory smoothness & (10) jerk trajectory smoothness
        accel_traj_smoothness = 0
        jerk_traj_smoothness = 0

        # (11) number of stops
        num_of_stops = 0

        ##
        ## Read the bag files
        ##

        bag_files = [f for f in os.listdir(os.path.join(DATA_DIR, sim_folder)) if f.endswith(".bag")]
        bag_files.sort()

        for bag_file in bag_files:

            print(bag_file)
            num_of_bags += 1
            agent_name = bag_file[8:13]
            topics = TOPICS_TO_UNPACK.format(*[agent_name for i in range(9)]).split(" ")

            with rosbag.Bag(os.path.join(DATA_DIR, sim_folder, bag_file)) as bag:

                ##
                ## (1) travel time
                ##

                sim_start_times = []
                sim_end_times = []

                for topic, msg, t in bag.read_messages(topics=topics):
                    if  topic == f"/{agent_name}/term_goal":
                        sim_start_times.append(msg.header.stamp.to_sec())
                    if topic == f"/sim_all_agents_goal_reached":
                        sim_end_times.append(msg.header.stamp.to_sec())

                ##
                ## (2) computation time
                ##

                computation_times = []

                for topic, msg, t in bag.read_messages(topics=topics):
                    if topic == f"/{agent_name}/panther/log":
                        computation_times.append(msg.ms_opt)

                ##
                ## (3) number of collisions (using tf)
                ##

                bag_transformer = BagTfTransformer(bag)
                sim_start_time = sim_start_times[-1]
                try: 
                    sim_end_time = sim_end_times[-1]
                except:
                    sim_end_time = bag.get_end_time()
                discrete_times = np.linspace(sim_start_time, sim_end_time, int((sim_end_time - sim_start_time) * 100))

                # get combination of an agent and an agent and an agent and an obstacle
                agent_agent_pairs = list(itertools.combinations(AGENTS_LIST, 2))
                # get a pair of an agent and an obstacle
                agent_obstacle_pairs = list(itertools.product(AGENTS_LIST, OBSTACLES_LIST))

                # check if the agent-agent pair is in collision
                for agent_agent_pair in agent_agent_pairs:
                    agent1 = agent_agent_pair[0]
                    agent2 = agent_agent_pair[1]
                    for t in discrete_times:
                        agent1_pose = bag_transformer.lookupTransform("world", agent1, rospy.Time.from_sec(t))
                        agent2_pose = bag_transformer.lookupTransform("world", agent2, rospy.Time.from_sec(t))
                        
                        x_diff = abs(agent1_pose[0][0] - agent2_pose[0][0])
                        y_diff = abs(agent1_pose[0][1] - agent2_pose[0][1])
                        z_diff = abs(agent1_pose[0][2] - agent2_pose[0][2])

                        if x_diff < BBOX_AGENT_AGENT[0] and y_diff < BBOX_AGENT_AGENT[1] and z_diff < BBOX_AGENT_AGENT[2]:
                            num_of_collisions_btwn_agents += 1
                            break
            
                # check if the agent-obstacle pair is in collision
                for agent_obstacle_pair in agent_obstacle_pairs:
                    agent = agent_obstacle_pair[0]
                    obstacle = agent_obstacle_pair[1]
                    for t in discrete_times:
                        agent_pose = bag_transformer.lookupTransform("world", agent, rospy.Time.from_sec(t))
                        obstacle_pose = bag_transformer.lookupTransform("world", obstacle, rospy.Time.from_sec(t))

                        x_diff = abs(agent1_pose[0][0] - agent2_pose[0][0])
                        y_diff = abs(agent1_pose[0][1] - agent2_pose[0][1])
                        z_diff = abs(agent1_pose[0][2] - agent2_pose[0][2])

                        if x_diff < BBOX_AGENT_OBST[0] and y_diff < BBOX_AGENT_OBST[1] and z_diff < BBOX_AGENT_OBST[2]:
                            num_of_collisions_btwn_agents_and_obstacles += 1
                            break
                
                ##
                ## (4) fov rate & (5) continuous fov detection
                ##

                flight_time = sim_end_time - sim_start_time
                discrete_times = np.linspace(sim_start_time, sim_end_time, int((sim_end_time - sim_start_time) * 100))
                dt = flight_time / len(discrete_times)

                for agent in AGENTS_LIST:
                    for obstacle in OBSTACLES_LIST:
                        is_in_FOV_in_prev_timestep = False
                        max_streak_in_FOV = 0
                        for t in discrete_times:

                            # check if the obstacle is in the FOV of the agent
                            agent_pos, agent_quat = bag_transformer.lookupTransform("world", agent, rospy.Time.from_sec(t))
                            obst_pos, _ = bag_transformer.lookupTransform("world", obstacle, rospy.Time.from_sec(t))
                            fov_rate[agent] += check_obst_is_in_FOV(agent_pos, agent_quat, obst_pos, FOV_X_DEG, FOV_Y_DEG, FOV_DEPTH)

                            # check if the agent is close to the obstacle
                            dist = np.linalg.norm(np.array(agent_pos) - np.array(obst_pos))
                            if dist < FOV_DEPTH:
                                agent_obstacle_proxy[agent] += 1

                            # check if the obstacle is in the FOV of the agent continuously
                            if check_obst_is_in_FOV(agent_pos, agent_quat, obst_pos, FOV_X_DEG, FOV_Y_DEG, FOV_DEPTH):
                                if not is_in_FOV_in_prev_timestep:
                                    max_streak_in_FOV = 1
                                else:
                                    max_streak_in_FOV += 1
                                    continuous_fov_detection[agent][obstacle] = max(continuous_fov_detection[agent][obstacle], max_streak_in_FOV)
                                is_in_FOV_in_prev_timestep = True
                            else:
                                is_in_FOV_in_prev_timestep = False

                # converet the count to rate
                for agent in AGENTS_LIST:
                    fov_rate[agent] = fov_rate[agent] / agent_obstacle_proxy[agent]
                
                ##
                ## (6) translational dynamic constraint violation rate & (7) yaw dynamic constraint violation rate
                ##

                topic_num = 0
                for topic, msg, t in bag.read_messages(topics=topics):

                    if  topic == f"/{agent_name}/goal":

                        vel = np.linalg.norm(np.array([msg.v.x, msg.v.y, msg.v.z]))
                        acc = np.linalg.norm(np.array([msg.a.x, msg.a.y, msg.a.z]))
                        jerk = np.linalg.norm(np.array([msg.j.x, msg.j.y, msg.j.z]))
                        dyaw = float(msg.dpsi)

                        if vel > math.sqrt(3)*MAX_VEL or acc > math.sqrt(3)*MAX_ACC or jerk > math.sqrt(3)*MAX_JERK:
                            translational_dynamic_constraint_violation_rate += 1
                        if dyaw > MAX_DYAW:
                            yaw_dynamic_constraint_violation_rate += 1
                        
                        topic_num += 1
                
                ##
                ## (9) accel trajectory smoothness & (10) jerk trajectory smoothness
                ##

                for topic, msg, t in bag.read_messages(topics=topics):

                    if  topic == f"/{agent_name}/goal":

                        acc = np.linalg.norm(np.array([msg.a.x, msg.a.y, msg.a.z]))
                        jerk = np.linalg.norm(np.array([msg.j.x, msg.j.y, msg.j.z]))

                        accel_traj_smoothness += acc
                        jerk_traj_smoothness += jerk

                ##
                ## (11) number of stops
                ##

                is_stopped = True
                for topic, msg, t in bag.read_messages(topics=topics):
                    if  topic == f"/{agent_name}/goal":

                        vel = np.linalg.norm(np.array([msg.v.x, msg.v.y, msg.v.z]))

                        if vel < 0.001 and not is_stopped:
                            num_of_stops += 1
                            is_stopped = True
                        elif vel > 0.001 and is_stopped:
                            is_stopped = False
                        else:
                            pass
                
                num_of_stops = num_of_stops - 1 # the last stop (when goal reached) is not counted

            ##
            ## Data extraction per bag
            ##

            # (1) travel time
            try:
                travel_time = max(sim_end_times) - min(sim_start_times)
            except:
                travel_time = bag.get_end_time() - min(sim_start_times)

            travel_time_list.append(travel_time)

            # (2) computation time
            computation_time_list.extend(computation_times) # right now I am not including octopus search time

            # (3) number of collisions
            num_of_collisions_btwn_agents_list.append(num_of_collisions_btwn_agents)
            num_of_collisions_btwn_agents_and_obstacles_list.append(num_of_collisions_btwn_agents_and_obstacles)

            # (4) fov rate
            for agent in AGENTS_LIST:
                fov_rate_list.append(fov_rate[agent])

            # (5) continuous fov detection (min, avg, max)
            for agent in AGENTS_LIST:
                for obstacle in OBSTACLES_LIST:
                    continuous_fov_detection_list.append(continuous_fov_detection[agent][obstacle])

            # (6) translational dynamic constraints violation rate
            translational_dynamic_constraint_violation_rate = translational_dynamic_constraint_violation_rate / topic_num
            translational_dynamic_constraint_violation_rate_list.append(translational_dynamic_constraint_violation_rate)

            # (7) yaw dynamic constraint violation rate
            yaw_dynamic_constraint_violation_rate = yaw_dynamic_constraint_violation_rate / topic_num
            yaw_dynamic_constraint_violation_rate_list.append(yaw_dynamic_constraint_violation_rate)

            # (8) success rate
            success = True if num_of_collisions_btwn_agents == 0 \
                and num_of_collisions_btwn_agents_and_obstacles == 0 \
                and translational_dynamic_constraint_violation_rate == 0.0 \
                and yaw_dynamic_constraint_violation_rate == 0.0 else False
            success_rate_list.append(success)

            # (9) accel trajectory smoothness & (10) jerk trajectory smoothness
            accel_traj_smoothness = accel_traj_smoothness * DT
            accel_trajectory_smoothness_list.append(accel_traj_smoothness)
            jerk_traj_smoothness = jerk_traj_smoothness * DT
            jerk_trajectory_smoothness_list.append(jerk_traj_smoothness)

            # (11) number of stops
            num_of_stops_list.append(num_of_stops)

        ##
        ## Data print per simulation environment (eg. 1_obs_1_agent)
        ##

        d_string     = f"date                                             {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        sf_string    = f"simulation folder                                {os.path.join(DATA_DIR, sim_folder)}"
        tt_string    = f"travel time                                      {round(mean(travel_time_list),2)} [s]"
        ct_string    = f"computational time                               {round(mean(computation_time_list),2)} [ms]"
        ncba_string  = f"number of collisions btwn agents                 {round(mean(num_of_collisions_btwn_agents_list),2)}"
        ncbao_string = f"number of collisions btwn agents and obstacles   {round(mean(num_of_collisions_btwn_agents_and_obstacles_list),2)}"
        fr_string    = f"fov rate                                         {round(mean(fov_rate_list),2)*100} [%]"
        cfd_string   = f"continuous fov detection                         {round(mean(continuous_fov_detection_list),2)}"
        tdcvr_string = f"translational dynamic constraint violation rate  {round(mean(translational_dynamic_constraint_violation_rate_list),2)*100} [%]"
        ydcvr_string = f"yaw dynamic constraint violation rate            {round(mean(yaw_dynamic_constraint_violation_rate_list),2)*100} [%]"
        sr_string    = f"success rate                                     {round(mean(success_rate_list),2)*100} [%]"
        ats_string   = f"accel trajectory smoothness                      {round(mean(accel_trajectory_smoothness_list),2)}"
        jts_string   = f"jerk trajectory smoothness                       {round(mean(jerk_trajectory_smoothness_list),2)}"
        ns_string    = f"number of stops                                  {round(mean(num_of_stops_list),2)}"
        
        print("\n")
        print("=============================================")
        print(sf_string)
        print(tt_string)
        print(ct_string)
        print(ncba_string)
        print(ncbao_string)
        print(fr_string)
        print(cfd_string)
        print(tdcvr_string)
        print(ydcvr_string)
        print(sr_string)
        print(ats_string)
        print(jts_string)
        print(ns_string)
        print("=============================================")

        ##
        ## Save the data
        ##

        with open(os.path.join(DATA_DIR, "data.txt"), "a") as f:
            f.write("\n")
            f.write("=============================================\n")
            f.write(d_string + "\n")
            f.write(sf_string + "\n")
            f.write(tt_string + "\n")
            f.write(ct_string + "\n")
            f.write(ncba_string + "\n")
            f.write(ncbao_string + "\n")
            f.write(fr_string + "\n")
            f.write(cfd_string + "\n")
            f.write(tdcvr_string + "\n")
            f.write(ydcvr_string + "\n")
            f.write(sr_string + "\n")
            f.write(ats_string + "\n")
            f.write(jts_string + "\n")
            f.write(ns_string + "\n")
            f.write("=============================================\n")