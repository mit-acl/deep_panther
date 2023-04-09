#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  *
#  * Input: home directory
#  * -------------------------------------------------------------------------- */

import math
import os
import sys
import time
import rospy
from snapstack_msgs.msg import State
import subprocess

def get_start_end_state(num_of_agents, circle_radius) -> (list, list, list, list, list, list, list):
    """
    get start state for the agent_id-th agent
    the agents will be placed in a circle with circle_radius radius, and the goal will be placed in the opposite direction
    the first agent will be placed at x=circle_radius, y=0, z =0, yaw=pi, and the rest will be placed in a counther clockwise fashion 
    """

    ## Note: the exactly same function is defined in panther/scripts/goal_reached_checker.py
    ## So if you change this function, you need to change the other one as well

    x_start_list = []
    y_start_list = []
    z_start_list = []
    yaw_start_list = []
    x_end_list = []
    y_end_list = []
    z_end_list = []

    for i in range(num_of_agents):
        angle = 2*math.pi/num_of_agents * i
        x_start_list.append(circle_radius*math.cos(angle))
        y_start_list.append(circle_radius*math.sin(angle))
        z_start_list.append(0.0)
        yaw_start_list.append(math.tan(y_start_list[i]/x_start_list[i]) + math.pi)
        x_end_list.append(-x_start_list[i])
        y_end_list.append(-y_start_list[i])
        z_end_list.append(0.0)

    return x_start_list, y_start_list, z_start_list, yaw_start_list, x_end_list, y_end_list, z_end_list

def check_goal_reached():
    try:
        is_goal_reached = subprocess.check_output(['rostopic', 'echo', '/sim_all_agents_goal_reached', '-n', '1'], timeout=2).decode()
        print("True")
        return True 
    except:
        print("False")
        return False  

if __name__ == '__main__':

    ##
    ## Parameters
    ##

    NUM_OF_SIMS = 100
    NUM_OF_AGENTS = 2
    NUM_OF_OBS_LIST = [5, 10, 15]
    CIRCLE_RADIUS = 10.0
    USE_PERFECT_CONTROLLER = "true"
    USE_PERFECT_PREDICTION = "true"
    SIM_DURATION = 60 # in seconds
    HOME_DIR = sys.args[1] if len(sys.argv) > 1 else "/media/kota/T7/deep-panther/bags"
    RECORD_NODE_NAME = "bag_recorder"
    KILL_ALL = "tmux kill-server & killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & killall -9  & killall -9 rosmaster & pkill panther_node & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f panther_commands"

    ##
    ## make sure ROS (and related stuff) is not running
    ##

    os.system(KILL_ALL)

    ##
    ## simulation loop
    ##

    for k in range(len(NUM_OF_OBS_LIST)):

        ##
        ## set up folders
        ##

        folder_bags = HOME_DIR + f"/{NUM_OF_OBS_LIST[k]}_obs"
        if not os.path.exists(folder_bags):
            os.makedirs(folder_bags)
        
        for s in range(NUM_OF_SIMS):

            ##
            ## commands list initialized
            ##

            commands = []

            time_sleep = max(0.2*NUM_OF_OBS_LIST[k], 2.0)
            time_sleep_goal = min(NUM_OF_OBS_LIST[k], NUM_OF_AGENTS, 10.0)

            ###
            ### simulation set up
            ###

            ## roscore
            commands.append("roscore")

            ## sim_basestation
            commands.append(f"roslaunch --wait panther sim_base_station.launch num_of_obs:={NUM_OF_OBS_LIST[k]} rviz:=true gui_mission:=false")

            ## sim_onboard
            x_start_list, y_start_list, z_start_list, yaw_start_list, x_end_list, y_end_list, z_end_list = get_start_end_state(NUM_OF_AGENTS, CIRCLE_RADIUS)
            for i, (x, y, z, yaw) in enumerate(zip(x_start_list, y_start_list, z_start_list, yaw_start_list)):
                agent_name = f"SQ{str(i+1).zfill(2)}s"
                commands.append(f"roslaunch --wait panther sim_onboard.launch quad:={agent_name} perfect_controller:={USE_PERFECT_CONTROLLER} perfect_prediction:={USE_PERFECT_PREDICTION} x:={x} y:={y} z:={z} yaw:={yaw}")
            
            ## rosbag record
            # commands.append("sleep "+str(time_sleep)+" && cd "+folder_bags+" && rosbag record -a -o sim_"+str(s)+" __name:="+RECORD_NODE_NAME)
            
            ## goal checker
            commands.append(f"sleep {time_sleep} && roslaunch --wait panther goal_reached_checker.launch num_of_agents:={NUM_OF_AGENTS} circle_radius:={CIRCLE_RADIUS}")

            ## publish goal
            for i, (x, y, z) in enumerate(zip(x_end_list, y_end_list, z_end_list)):
                agent_name = f"SQ{str(i+1).zfill(2)}s"
                ## TODO: may need to change the goal orientation
                commands.append(f"sleep "+str(time_sleep_goal)+" && rostopic pub /"+agent_name+"/term_goal geometry_msgs/PoseStamped \'{header: {stamp: now, frame_id: \"world\"}, \
                    pose: {position: {x: "+str(x)+", y: "+str(y)+", z: "+str(z)+"}, orientation: {w: 1.0}}}\'")

            ##
            ## tmux & sending commands
            ##

            session_name="run_many_sims_multiagent_session"
            os.system("tmux kill-session -t" + session_name)
            os.system("tmux new -d -s "+str(session_name)+" -x 300 -y 300")

            for i in range(len(commands)):
                os.system('tmux split-window ; tmux select-layout tiled')
            
            for i in range(len(commands)):
                os.system('tmux send-keys -t '+str(session_name)+':0.'+str(i) +' "'+ commands[i]+'" '+' C-m')

            print("commands sent")
            time.sleep(3.0)

            ##
            ## wait until the goal is reached
            ##

            is_goal_reached = False
            tic = time.perf_counter()
            toc = time.perf_counter()

            while (toc - tic < SIM_DURATION and not is_goal_reached):
                toc = time.perf_counter()
                if(check_goal_reached()):
                    print('all the agents reached the goal')
                    is_goal_reached = True
                time.sleep(0.1)

            if (not is_goal_reached):
                os.system(f'echo "simulation {s}: not goal reached" >> '+folder_bags+'/status.txt')
                print("Goal is not reached, killing the bag node")
            else:
                os.system(f'echo "simulation {s}: goal reached" >> '+folder_bags+'/status.txt')
                print("Goal is reached, killing the bag node")

            os.system("rosnode kill "+RECORD_NODE_NAME)
            time.sleep(0.5)
            print("Killing the rest")
            os.system(KILL_ALL)

    time.sleep(3.0)