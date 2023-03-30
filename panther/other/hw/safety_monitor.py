#Author: Kota Kondo
#Date: March 28, 2023

#--------------------------------------------------------------------------------------------------
# This node monitors the safety of the swarm. It sends a motor-kill command when safety is violated.
# 
# input: agent names
# output: motor-kill command when safety is violated
#
# TODO: right now we have a maximum agent number of 6. We need to make this dynamic.
#--------------------------------------------------------------------------------------------------

import math
import os
import sys
import time
import rospy
import rosgraph
import numpy as np

from geometry_msgs.msg import PoseStamped
from snapstack_msgs.msg import State, AttitudeCommand

from behavior_selector.srv import MissionModeChange

class SafetyMonitor:

    def __init__(self, agent_names):

        ##
        ## get space boundaries (TODL: it's hard-coded -- make this dynamic)
        ##

        self.x_max = 4.0
        self.x_min = -4.0
        self.y_max = 4.0
        self.y_min = -3.0
        self.z_max = 3.0
        self.z_min = 1.0

        ##
        ## initializatoin
        ##

        self.agent_names = agent_names
        self.num_of_agents = len(agent_names)
        self.initialized = False
        self.initialized_mat = [False for i in range(self.num_of_agents)]
        self.state_pos = np.empty([self.num_of_agents,3])
        self.killed = False

        ##
        ## publisher for motor kill
        ##

        self.publishers = []
        for i, agent_name in enumerate(agent_names):
            # define publisher for each agent
            rospy.wait_for_service('change_mode',1)
            self.publishers.append(rospy.ServiceProxy('change_mode', MissionModeChange))

    def monitor_safety(self, timer):
        
        ##
        ## check if all agents are initialized
        ##

        if not self.initialized:
            initialized = True
            for k in range(self.num_of_agents):
                if not self.initialized_mat[k]:
                    initialized = False
                    break
            self.initialized = initialized

        ##
        ## check safety
        ##

        if self.initialized:
            for i, agent_name in enumerate(self.agent_names):
                x, y, z = self.state_pos[i,0:3]
                if x > self.x_max or x < self.x_min or \
                    y > self.y_max or y < self.y_min or \
                        z > self.z_max or z < self.z_min:

                        ##
                        ## kill motors
                        ##

                        if not self.killed:
                            print(f"[{agent_name}] safety is violated!")
                            print(f"current position: {x}, {y}, {z}")
                            KILL = 3
                            self.publishers[i](mode=KILL)
                            self.killed = True
    ##
    ## callback functions for each agent
    ##

    def Agent01stateCB(self, data):
        self.state_pos[0,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
        if self.initialized_mat[0] == False and self.state_pos[0,2] > self.z_min: # make sure first [0, 0, 0] state info will not be used
            self.initialized_mat[0] = True
    def Agent02stateCB(self, data):
        self.state_pos[1,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
        if self.initialized_mat[1] == False and self.state_pos[1,2] > self.z_min: # make sure first [0, 0, 0] state info will not be used
            self.initialized_mat[1] = True
    def Agent03stateCB(self, data):
        self.state_pos[2,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
        if self.initialized_mat[2] == False and self.state_pos[2,2] > self.z_min: # make sure first [0, 0, 0] state info will not be used
            self.initialized_mat[2] = True
    def Agent04stateCB(self, data):
        self.state_pos[3,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
        if self.initialized_mat[3] == False and self.state_pos[3,2] > self.z_min: # make sure first [0, 0, 0] state info will not be used
            self.initialized_mat[3] = True
    def Agent05stateCB(self, data):
        self.state_pos[4,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
        if self.initialized_mat[4] == False and self.state_pos[4,2] > self.z_min: # make sure first [0, 0, 0] state info will not be used
            self.initialized_mat[4] = True
    def Agent06stateCB(self, data):
        self.state_pos[5,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])
        if self.initialized_mat[5] == False and self.state_pos[5,2] > self.z_min: # make sure first [0, 0, 0] state info will not be used
            self.initialized_mat[5] = True

def startNode(agent_names):

    ##
    ## get class
    ##

    c = SafetyMonitor(agent_names=agent_names)

    ##
    ## subscribe to each agent (max 6 agents)
    ##

    for i, agent_name in enumerate(agent_names):
        func = getattr(c, f"Agent0{i+1}stateCB")
        rospy.Subscriber(f"{agent_name}/world", State, func)
        # rospy.Subscriber(f"{agent_name}/state", State, func)
    
    ##
    ## start timer and spin
    ##

    rospy.Timer(rospy.Duration(0.001), c.monitor_safety)
    rospy.spin()

if __name__ == '__main__':

    ##
    ## get agent names
    ##

    num_agents = len(sys.argv) - 1
    if num_agents < 1:
        print("Usage: python SafetyMonitor.py [agent1_name, agent2_name, ...]")
        sys.exit()
    
    agent_names = []
    for i in range(1, num_agents+1):
        agent_names.append(sys.argv[i])

    ##
    ## initialize and start node
    ##

    rospy.init_node('SafetyMonitor')
    startNode(agent_names)