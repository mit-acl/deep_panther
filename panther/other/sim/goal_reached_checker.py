#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  * -------------------------------------------------------------------------- */

import math
import os
import sys
import time
import rospy
import rosgraph
from geometry_msgs.msg import PoseStamped
from snapstack_msgs.msg import State
from panther_msgs.msg import GoalReached
import numpy as np
from random import *
import tf2_ros
from numpy import linalg as LA
from run_many_sims import get_start_end_state

class GoalReachedCheck:

    def __init__(self, num_of_agents, circle_radius):

        rospy.sleep(3)

        # goal radius
        self.goal_radius = 0.5 #needs to be the same as the one in panther.yaml

        # number of agents
        self.num_of_agents = num_of_agents

        # is initialized?
        self.initialized = False

        # state and term_goal
        self.state_pos = np.empty([self.num_of_agents,3])
        self.term_goal_pos = np.empty([self.num_of_agents,3])

        # publisher init
        self.goal_reached = GoalReached()
        self.pubIsGoalReached = rospy.Publisher('/sim_all_agents_goal_reached', GoalReached, queue_size=1, latch=True)

        # is goal reached
        self.is_goal_reached = False

        # keep track of which drone has already got to the goal
        self.is_goal_reached_check_list = [False for i in range(self.num_of_agents)]

    # goal reached checker
    def goal_reached_checker(self, timer):
        if not self.is_goal_reached and self.initialized:
            for i in range(self.num_of_agents):
                if self.is_goal_reached_check_list[i] == False:
                    if (LA.norm(self.state_pos[i,:] - self.term_goal_pos[i,:]) > self.goal_radius):
                        return
                    else:
                        if self.is_goal_reached_check_list[i] == False:
                            self.is_goal_reached_check_list[i] = True

                        if self.is_goal_reached_check_list.count(True) == self.num_of_agents:
                            self.is_goal_reached = True
                            now = rospy.get_rostime()
                            self.goal_reached.header.stamp = now
                            self.goal_reached.is_goal_reached = True
                            self.pubIsGoalReached.publish(self.goal_reached)

    def SQ01stateCB(self, data):
        self.state_pos[0,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    def SQ02stateCB(self, data):
        self.state_pos[1,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    def SQ03stateCB(self, data):
        self.state_pos[2,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    def SQ04stateCB(self, data):
        self.state_pos[3,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    def SQ05stateCB(self, data):
        self.state_pos[4,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    def SQ06stateCB(self, data):
        self.state_pos[5,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    def SQ07stateCB(self, data):
        self.state_pos[6,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    def SQ08stateCB(self, data):
        self.state_pos[7,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    def SQ09stateCB(self, data):
        self.state_pos[8,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    def SQ10stateCB(self, data):
        self.state_pos[9,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    # def SQ11stateCB(self, data):
    #     self.state_pos[10,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    # def SQ12stateCB(self, data):
    #     self.state_pos[11,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    # def SQ13stateCB(self, data):
    #     self.state_pos[12,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    # def SQ14stateCB(self, data):
    #     self.state_pos[13,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    # def SQ15stateCB(self, data):
    #     self.state_pos[14,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
    # def SQ16stateCB(self, data):
    #     self.state_pos[15,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])


    def SQ01term_goalCB(self, data):
        self.term_goal_pos[0,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.initialized = True
    def SQ02term_goalCB(self, data):
        self.term_goal_pos[1,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    def SQ03term_goalCB(self, data):
        self.term_goal_pos[2,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    def SQ04term_goalCB(self, data):
        self.term_goal_pos[3,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    def SQ05term_goalCB(self, data):
        self.term_goal_pos[4,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    def SQ06term_goalCB(self, data):
        self.term_goal_pos[5,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    def SQ07term_goalCB(self, data):
        self.term_goal_pos[6,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    def SQ08term_goalCB(self, data):
        self.term_goal_pos[7,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    def SQ09term_goalCB(self, data):
        self.term_goal_pos[8,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    def SQ10term_goalCB(self, data):
        self.term_goal_pos[9,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    # def SQ11term_goalCB(self, data):
    #     self.term_goal_pos[10,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    # def SQ12term_goalCB(self, data):
    #     self.term_goal_pos[11,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    # def SQ13term_goalCB(self, data):
    #     self.term_goal_pos[12,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    # def SQ14term_goalCB(self, data):
    #     self.term_goal_pos[13,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    # def SQ15term_goalCB(self, data):
    #     self.term_goal_pos[14,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    # def SQ16term_goalCB(self, data):
    #     self.term_goal_pos[15,0:3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])

def startNode(num_agents, circle_radius):

    c = GoalReachedCheck(num_agents, circle_radius)

    ## Subscribe to the state of each agent and the terminal goal
    for i in range(num_agents):
        call_back_function = getattr(c, "SQ%02dstateCB" % (i+1))
        rospy.Subscriber("SQ%02ds/state" % (i+1), State, call_back_function)
        call_back_function = getattr(c, "SQ%02dterm_goalCB" % (i+1))
        rospy.Subscriber("SQ%02ds/term_goal" % (i+1), PoseStamped, call_back_function)
    
    rospy.Timer(rospy.Duration(1.0), c.goal_reached_checker)
    rospy.spin()

if __name__ == '__main__':

    ##
    ## get params
    ##

    num_of_agents = rospy.get_param("goal_reached_checker/num_of_agents")
    circle_radius = rospy.get_param("goal_reached_checker/circle_radius")

    rospy.init_node('goalReachedCheck')
    startNode(num_of_agents, circle_radius)