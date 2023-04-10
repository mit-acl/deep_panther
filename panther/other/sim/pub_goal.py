#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  *
#  * Publish goal to all the agents when they are all ready
#  * -------------------------------------------------------------------------- */

import math
import os
import sys
import time
import rospy
import rosgraph
from geometry_msgs.msg import PoseStamped
from snapstack_msgs.msg import State
from panther_msgs.msg import IsReady
import numpy as np
from random import *
import tf2_ros
from numpy import linalg as LA
from run_many_sims import get_start_end_state

class ReadyCheck:

    def __init__(self, x_goal_list, y_goal_list, z_goal_list):

        rospy.sleep(3)

        # goal radius
        self.goal_radius = 0.5 #needs to be the same as the one in panther.yaml

        # number of agents
        assert len(x_goal_list) == len(y_goal_list) == len(z_goal_list)
        self.num_of_agents = len(x_goal_list)

        # goal lists
        self.x_goal_list = x_goal_list
        self.y_goal_list = y_goal_list
        self.z_goal_list = z_goal_list

        # is_ready_cnt
        self.is_ready_cnt = 0

    # goal reached checker
    def is_ready_checker(self, timer):
        if self.is_ready_cnt == self.num_of_agents:
            ## publish goal
            commands = []
            for i, (x, y, z) in enumerate(zip(self.x_goal_list, self.y_goal_list, self.z_goal_list)):
                agent_name = f"SQ{str(i+1).zfill(2)}s"
                ## TODO: may need to change the goal orientation
                commands.append("rostopic pub /"+agent_name+"/term_goal geometry_msgs/PoseStamped \'{header: {stamp: now, frame_id: \"world\"}, \
                    pose: {position: {x: "+str(x)+", y: "+str(y)+", z: "+str(z)+"}, orientation: {w: 1.0}}}\'")
            
            ## send goals to all the agents
            session_name = "pub_goal"
            os.system("tmux kill-session -t" + session_name)
            os.system("tmux new-session -d -s "+str(session_name)+" -x 300 -y 300")
            time.sleep(1.0)
            # tmux splitting
            for i in range(len(commands)):
                os.system('tmux new-window -t ' + str(session_name))
            time.sleep(1.0)
            for i in range(len(commands)):
                os.system('tmux send-keys -t '+str(session_name)+':'+str(i) +'.0 "'+ commands[i]+'" '+' C-m')

            ## shutdown the node
            rospy.signal_shutdown("All the agents are ready")

    def SQ01s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ02s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ03s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ04s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ05s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ06s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ07s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ08s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ09s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ10s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ11s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ12s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ13s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ14s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ15s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1
    def SQ16s_is_readyCB(self, data):
        self.is_ready_cnt = self.is_ready_cnt + 1

def startNode(x_goal_list, y_goal_list, z_goal_list):

    c = ReadyCheck(x_goal_list, y_goal_list, z_goal_list)

    ## Subscribe to IsReady of each agent
    assert len(x_goal_list) == len(y_goal_list) == len(z_goal_list)
    for i in range(len(x_goal_list)):
        call_back_function = getattr(c, "SQ%02ds_is_readyCB" % (i+1))
        rospy.Subscriber("/SQ%02ds/panther/is_ready" % (i+1), IsReady, call_back_function)
    
    rospy.Timer(rospy.Duration(1.0), c.is_ready_checker)
    rospy.spin()

if __name__ == '__main__':

    ##
    ## get params
    ##

    x_goal_list = rospy.get_param("/pub_goal/x_goal_list")
    y_goal_list = rospy.get_param("/pub_goal/y_goal_list")
    z_goal_list = rospy.get_param("/pub_goal/z_goal_list")

    print("x_goal_list: ", x_goal_list)
    print("y_goal_list: ", y_goal_list)
    print("z_goal_list: ", z_goal_list)

    rospy.init_node('sendGoal')
    startNode(x_goal_list, y_goal_list, z_goal_list)