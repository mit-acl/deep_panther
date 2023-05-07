#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  *
#  * Input: directory to save bags (if not given the default is  "/media/kota/T7/deep-panther/bags")
#  * Output: bags with the name "sim_{sim_number}_{agent_name}_{date}.bag"
#  * -------------------------------------------------------------------------- */

import math
import os
import sys
import time
import rospy
from snapstack_msgs.msg import State
import subprocess

def get_start_end_state(num_of_agents, circle_radius, INITIAL_POSITIONS_SHAPE) -> (list, list, list, list, list, list, list):
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
    x_goal_list = []
    y_goal_list = []
    z_goal_list = []

    if INITIAL_POSITIONS_SHAPE=="circle":

        for i in range(num_of_agents):
            angle = 2*math.pi/num_of_agents * i
            x_start_list.append(circle_radius*math.cos(angle))
            y_start_list.append(circle_radius*math.sin(angle))
            z_start_list.append(0.0)
            yaw_start_list.append(math.atan2(y_start_list[i], x_start_list[i]) + math.pi)
            x_goal_list.append(-x_start_list[i])
            y_goal_list.append(-y_start_list[i])
            z_goal_list.append(0.0)

    elif INITIAL_POSITIONS_SHAPE=="square": #square is supported up to 4 agents

        for i in range(num_of_agents):
            angle = math.pi/2 * i + math.pi/4
            x_start_list.append(circle_radius*math.cos(angle))
            y_start_list.append(circle_radius*math.sin(angle))
            z_start_list.append(0.0)
            yaw_start_list.append(math.atan2(y_start_list[i], x_start_list[i]) + math.pi)
            x_goal_list.append(-x_start_list[i])
            y_goal_list.append(-y_start_list[i])
            z_goal_list.append(0.0)

    return x_start_list, y_start_list, z_start_list, yaw_start_list, x_goal_list, y_goal_list, z_goal_list

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
    NUM_OF_AGENTS = [1]
    NUM_OF_OBS_LIST = [2]
    CIRCLE_RADIUS = 5.0
    USE_PERFECT_CONTROLLER = "true"
    USE_PERFECT_PREDICTION = "true"
    SIM_DURATION = 60 # in seconds
    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/media/kota/T7/deep-panther/bags"
    RECORD_NODE_NAME = "bag_recorder"
    KILL_ALL = "killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f panther & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill rmader_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f rmader_commands & pkill -f dynamic_corridor & tmux kill-server & pkill -f perfect_controller & pkill -f publish_in_gazebo"
    TOPICS_TO_RECORD = "/{}/goal /{}/state /tf /tf_static /{}/panther/fov /obstacles_mesh /{}/panther/best_solution_expert /{}/panther/best_solution_student /{}/term_goal /{}/panther/actual_traj /clock /trajs /sim_all_agents_goal_reached /{}/panther/is_ready /{}/panther/log"
    USE_RVIZ = sys.argv[2] if len(sys.argv) >2 else "true"
    AGENTS_TYPES = ["parm_star"]
    TRAJ_NUM_PER_REPLAN_LIST = [10]
    DEFAULT_NUM_MAX_OF_OBST = 2 #TODO: hard-coded
    PRIMER_NUM_MAX_OF_OBST = 2
    INITIAL_POSITIONS_SHAPE = "circle" #circle or square (square is up to 4 agents)
    
    ##
    ## make sure ROS (and related stuff) is not running
    ##

    os.system(KILL_ALL)

    ##
    ## comment out some parameters in panther.yaml to overwrite them
    ##

    os.system("sed -i '/use_panther_star:/s/^/#/g' $(rospack find panther)/param/panther.yaml")
    os.system("sed -i '/use_expert:/s/^/#/g' $(rospack find panther)/param/panther.yaml") ## added : to avoid commenting out use_expert_for_other_agents_in_training
    os.system("sed -i '/use_student:/s/^/#/g' $(rospack find panther)/param/panther.yaml")
    # os.system("sed -i '/num_of_trajs_per_replan:/s/^/#/g' $(rospack find panther)/param/panther.yaml")
    # os.system("sed -i '/max_num_of_initial_guesses:/s/^/#/g' $(rospack find panther)/param/panther.yaml")
    # os.system("sed -i '/num_max_of_obst:/s/^/#/g' $(rospack find panther)/matlab/casadi_generated_files/params_casadi.yaml")

    ##
    ## simulation loop
    ##

    for traj_num in TRAJ_NUM_PER_REPLAN_LIST:

        for agent_type in AGENTS_TYPES:

            if traj_num != 10 and agent_type == "primer":
                # primer always produce 6 trajs as the NN's output is fixed
                continue

            DATA_AGENT_TYPE_DIR = DATA_DIR + f"/{traj_num}_traj/{agent_type}"

            for k in range(len(NUM_OF_OBS_LIST)):

                for l in NUM_OF_AGENTS:

                    ##
                    ## set up folders
                    ##

                    folder_bags = DATA_AGENT_TYPE_DIR + f"/{NUM_OF_OBS_LIST[k]}_obs_{l}_agents"
                    if not os.path.exists(folder_bags):
                        os.makedirs(folder_bags)
                    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    os.system(f'echo "\n{date}" >> '+folder_bags+'/status.txt')

                
                    for s in range(NUM_OF_SIMS):

                        ##
                        ## commands list initialized
                        ##

                        commands = []
                        time_sleep = max(0.2*NUM_OF_OBS_LIST[k], 2.0)
                        time_sleep_goal = min(NUM_OF_OBS_LIST[k], l, 2.0)

                        ##
                        ## simulation set up
                        ##

                        ## roscore
                        commands.append("roscore")

                        ## sim_basestation
                        commands.append(f"roslaunch --wait panther sim_base_station.launch num_of_obs:={NUM_OF_OBS_LIST[k]} rviz:={USE_RVIZ} gui_mission:=false")
                        
                        ## set up parameters depending on agent types
                        for i in range(l):
                            agent_name = f"SQ{str(i+1).zfill(2)}s"
                            if agent_type == "parm":
                                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/use_panther_star false")
                                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/use_expert true")
                                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/use_student false")
                                # commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/num_of_trajs_per_replan {traj_num}")
                                # commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/max_num_of_initial_guesses {traj_num}")
                                # commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/num_max_of_obst {DEFAULT_NUM_MAX_OF_OBST}")
                            elif agent_type == "parm_star":
                                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/use_panther_star true")
                                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/use_expert true")
                                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/use_student false")
                                # commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/num_of_trajs_per_replan {traj_num}")
                                # commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/max_num_of_initial_guesses {traj_num}")
                                # commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/num_max_of_obst {DEFAULT_NUM_MAX_OF_OBST}")
                            elif agent_type == "primer":
                                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/use_panther_star true")
                                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/use_expert false")
                                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/use_student true")
                                # commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/num_of_trajs_per_replan {traj_num}")
                                # commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/max_num_of_initial_guesses {traj_num}")
                                # commands.append(f"sleep 2.0 && rosparam set /{agent_name}/panther/num_max_of_obst {PRIMER_NUM_MAX_OF_OBST}")
                                

                        ## sim_onboard
                        x_start_list, y_start_list, z_start_list, yaw_start_list, x_goal_list, y_goal_list, z_goal_list = get_start_end_state(l, CIRCLE_RADIUS, INITIAL_POSITIONS_SHAPE)
                        for i, (x, y, z, yaw) in enumerate(zip(x_start_list, y_start_list, z_start_list, yaw_start_list)):
                            agent_name = f"SQ{str(i+1).zfill(2)}s"
                            commands.append(f"sleep 5.0 && roslaunch --wait panther sim_onboard.launch quad:={agent_name} perfect_controller:={USE_PERFECT_CONTROLLER} perfect_prediction:={USE_PERFECT_PREDICTION} x:={x} y:={y} z:={z} yaw:={yaw} 2> >(grep -v -e TF_REPEATED_DATA -e buffer)")
                        
                        ## rosbag record
                        # agent_bag_recorders = []
                        # for i in range(l):
                        #     sim_name = f"sim_{str(s).zfill(3)}"
                        #     agent_name = f"SQ{str(i+1).zfill(2)}s"
                        #     recorded_topics = TOPICS_TO_RECORD.format(*[agent_name for i in range(9)])
                        #     agent_bag_recorder = f"{agent_name}_{RECORD_NODE_NAME}"
                        #     agent_bag_recorders.append(agent_bag_recorder)
                        #     commands.append("sleep "+str(time_sleep)+" && cd "+folder_bags+" && rosbag record "+recorded_topics+" -o "+sim_name+"_"+agent_name+" __name:="+agent_bag_recorder)

                        recorded_topics = ""
                        for i in range(l):
                            recorded_topics += TOPICS_TO_RECORD.format(*[agent_name for i in range(9)])

                        sim_name = f"sim_{str(s).zfill(3)}"
                        sim_bag_recorder = sim_name
                        commands.append('sleep '+str(time_sleep)+' && cd '+folder_bags+' && rosbag record '+recorded_topics+' -o '+sim_name+' __name:='+sim_bag_recorder)
                        
                        ## goal checker
                        commands.append(f"sleep {time_sleep} && roslaunch --wait panther goal_reached_checker.launch num_of_agents:={l} circle_radius:={CIRCLE_RADIUS}")

                        ## publish goal
                        commands.append(f"sleep "+str(time_sleep_goal)+f" && roslaunch --wait panther pub_goal.launch x_goal_list:=\"{x_goal_list}\" y_goal_list:=\"{y_goal_list}\" z_goal_list:=\"{z_goal_list}\"")

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

                        os.system("rosnode kill "+sim_bag_recorder)
                        time.sleep(0.5)
                        print("Killing the rest")
                        os.system(KILL_ALL)

                        time.sleep(3.0)

    ## 
    ## uncomment delay_check param
    ##

    os.system("sed -i '/use_panther_star:/s/^#//g' $(rospack find panther)/param/panther.yaml")
    os.system("sed -i '/use_expert:/s/^#//g' $(rospack find panther)/param/panther.yaml") ## added : to avoid commenting out use_expert_for_other_agents_in_training
    os.system("sed -i '/use_student:/s/^#//g' $(rospack find panther)/param/panther.yaml")
    os.system("sed -i '/num_of_trajs_per_replan:/s/^#//g' $(rospack find panther)/param/panther.yaml")
    os.system("sed -i '/max_num_of_initial_guesses:/s/^#//g' $(rospack find panther)/param/panther.yaml")
    # os.system("sed -i '/num_max_of_obst:/s/^#//g' $(rospack find panther)/matlab/casadi_generated_files/params_casadi.yaml")
    
