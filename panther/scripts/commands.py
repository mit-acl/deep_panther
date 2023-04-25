#!/usr/bin/env python

# /* ----------------------------------------------------------------------------
#  * Copyright 2020, Jesus Tordesillas Torres, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  * -------------------------------------------------------------------------- */

import rospy
from panther_msgs.msg import WhoPlans
from snapstack_msgs.msg import Goal, State
from geometry_msgs.msg import Pose, PoseStamped
from snapstack_msgs.msg import QuadFlightMode
#from behavior_selector.srv import MissionModeChange
import math
import sys
import numpy as np

def quat2yaw(q):
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))
    return yaw

class PantherCommands:

    def __init__(self):
        self.whoplans=WhoPlans()
        self.pose = Pose()
        self.whoplans.value=self.whoplans.OTHER
        self.pubGoal = rospy.Publisher('goal', Goal, queue_size=1)
        self.pubWhoPlans = rospy.Publisher("who_plans",WhoPlans,queue_size=1,latch=True) 
        self.timer_take_off=rospy.Timer(rospy.Duration(0.004), self.timerTakeOffCB)
        self.timer_take_off.shutdown()
        
        self.alt_taken_off = 2.0; #Altitude when hovering after taking off
        self.initialized = False
        self.is_killed = False
        self.takeoff_goal=Goal()

    #In rospy, the callbacks are all of them in separate threads
    def stateCB(self, data):
        self.pose.position.x = data.pos.x
        self.pose.position.y = data.pos.y
        self.pose.position.z = data.pos.z
        self.pose.orientation = data.quat

        if self.initialized == False:
            self.init_pos = np.array([data.pos.x, data.pos.y, data.pos.z])
            self.initialized = True

    #Called when buttom pressed in the interface
    def globalflightmodeCB(self,req):
        if self.initialized == False:
            print ("Not initialized yet. Is DRONE_NAME/state being published?")
            return

        if req.mode == req.GO and self.whoplans.value==self.whoplans.OTHER:
            print ("Starting taking off")
            self.takeOff()
            print ("Take off done")
            self.whoplans.value=self.whoplans.PANTHER

        if req.mode == req.KILL:
            self.timer_take_off.shutdown()
            print ("Killing")
            self.kill()
            print ("Killed done")

        if req.mode == req.LAND and self.whoplans.value==self.whoplans.PANTHER:
            self.timer_take_off.shutdown()
            print ("Landing")
            self.land()
            print ("Landing done")

    def sendWhoPlans(self):
        self.whoplans.header.stamp = rospy.get_rostime()
        self.pubWhoPlans.publish(self.whoplans)

    def takeOff(self):
        self.is_killed = False
        self.takeoff_goal.p.x = self.pose.position.x
        self.takeoff_goal.p.y = self.pose.position.y
        self.takeoff_goal.p.z = self.pose.position.z
        self.takeoff_goal.psi = quat2yaw(self.pose.orientation)
        self.takeoff_goal.power= True #Turn on the motors

        #Note that self.pose.position is being updated in the parallel callback
        self.timer_take_off=rospy.Timer(rospy.Duration(0.002), self.timerTakeOffCB)

    def timerTakeOffCB(self, event):
        self.takeoff_goal.p.z = min(self.takeoff_goal.p.z+0.0005, self.alt_taken_off)
        rospy.loginfo_throttle(0.5, "Taking off..., error={}".format(self.pose.position.z-self.alt_taken_off))
        self.sendGoal(self.takeoff_goal)

        threshhold = 0.1
        if abs(self.pose.position.z-self.alt_taken_off) < threshhold:
            self.timer_take_off.shutdown()
            self.whoplans.value=self.whoplans.PANTHER
            self.sendWhoPlans()

    def land(self):
        self.is_killed = False
        self.whoplans.value=self.whoplans.OTHER
        self.sendWhoPlans()

        goal=Goal()
        goal.p.x = self.pose.position.x
        goal.p.y = self.pose.position.y
        goal.p.z = self.pose.position.z
        goal.power= True #Motors still on
        goal.psi = quat2yaw(self.pose.orientation)

        #Note that self.pose.position is being updated in the parallel callback
        while(abs(self.pose.position.z-self.init_pos[2])>0.08):
            goal.p.z = max(goal.p.z-0.0035, self.init_pos[2])
            rospy.sleep(0.01)
            self.sendGoal(goal)
        
        #Kill motors once we are on the ground
        self.kill()

    def kill(self):
        self.is_killed = True
        self.whoplans.value=self.whoplans.OTHER
        self.sendWhoPlans()
        goal=Goal()
        goal.p.x = self.pose.position.x
        goal.p.y = self.pose.position.y
        goal.p.z = self.pose.position.z
        goal.psi = quat2yaw(self.pose.orientation)
        goal.power=False #Turn off the motors 
        self.sendGoal(goal) #TODO: due to race conditions, publishing whoplans.OTHER and then goal.power=False does NOT guarantee that the external planner doesn't publish a goal with power=true
                            #The easy workaround is to click several times in the 'kill' button of the GUI

    def sendGoal(self, goal):
        goal.header.stamp = rospy.get_rostime()
        self.pubGoal.publish(goal)
                  
def startNode():
    c = PantherCommands()
    rospy.Subscriber("state", State, c.stateCB)
    rospy.Subscriber("/globalflightmode", QuadFlightMode, c.globalflightmodeCB)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('panther_commands')  
    startNode()
    print ("Behavior selector started")