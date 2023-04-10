#!/usr/bin/env python

import math
import os
import time
import rospy
import rosgraph
from geometry_msgs.msg import PoseStamped
from snapstack_msgs.msg import State
import numpy as np
from random import * 

class TermGoalSender:

    def __init__(self):

        # home yet?
        self.is_home = False

        # position change
        self.sign = 1

        # initialization done?
        self.is_init_pos = False

        # reached goal?
        self.if_arrived = False

        # term_goal init
        self.term_goal=PoseStamped()
        self.term_goal.header.frame_id='world'
        self.pubTermGoal = rospy.Publisher('term_goal', PoseStamped, queue_size=1, latch=True)
        
        # state_pos init ()
        self.state_pos=np.array([0.0, 0.0, 0.0])

        # waypoints
        self.wpidx = 0
        self.wps = np.array([
                            [-3.0, 3.0, 2.0],
                            [3.0, -3.0, 2.0]
                            ])

        # every 10 sec change goals
        rospy.Timer(rospy.Duration(10.0), self.change_goal)

        # set initial time and how long the demo is
        self.time_init = rospy.get_rostime()
        self.total_secs = 60.0; # sec

        # every 0.01 sec timerCB is called back
        self.is_change_goal = True
        self.timer = rospy.Timer(rospy.Duration(0.01), self.timerCB)

        # send goal
        self.sendGoal()

    def change_goal(self, tmp):
        self.is_change_goal = True
        

    def timerCB(self, tmp):
        
        # check if we should go home
        duration = rospy.get_rostime() - self.time_init
        if (duration.to_sec() > self.total_secs and not self.is_home):
            self.is_home = True
            self.sendGoal()

        # term_goal in array form
        self.term_goal_pos=np.array([self.term_goal.pose.position.x,self.term_goal.pose.position.y,self.term_goal.pose.position.z])

        # distance
        dist=np.linalg.norm(self.term_goal_pos-self.state_pos)
        #print("dist=", dist)

        # check distance and if it's close enough publish new term_goal
        dist_limit = 0.5
        if (dist < dist_limit):
            if not self.is_home:
                self.sendGoal()

    def sendGoal(self):

        if self.is_home:
            
            print ("Home Return")
            # set home goals
            self.term_goal.pose.position.x = self.init_pos[0]
            self.term_goal.pose.position.y = self.init_pos[1]
            self.term_goal.pose.position.z = 1.8

        else: 

            # set goals (exact position exchange, this could lead to drones going to exact same locations)
            self.term_goal.pose.position.x = self.wps[self.wpidx % 2][0]
            self.term_goal.pose.position.y = self.wps[self.wpidx % 2][1]
            self.term_goal.pose.position.z = self.wps[self.wpidx % 2][2]    

            self.if_arrived = not self.if_arrived
            self.wpidx += 1

        self.pubTermGoal.publish(self.term_goal)

        return

    def stateCB(self, data):
        if not self.is_init_pos:
            self.init_pos = np.array([data.pos.x, data.pos.y, data.pos.z])
            self.is_init_pos = True

        self.state_pos = np.array([data.pos.x, data.pos.y, data.pos.z])

def startNode():
    c = TermGoalSender()
    rospy.Subscriber("state", State, c.stateCB)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('TermGoalSender')
    startNode()