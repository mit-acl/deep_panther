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

def quat2yaw(q):
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))
    return yaw

class Panther_Commands:

    def __init__(self):
        self.whoplans=WhoPlans();
        self.pose = Pose();
        self.whoplans.value=self.whoplans.OTHER
        self.pubGoal = rospy.Publisher('goal', Goal, queue_size=1)
        self.pubWhoPlans = rospy.Publisher("who_plans",WhoPlans,queue_size=1,latch=True) 
        #self.pubClickedPoint = rospy.Publisher("/move_base_simple/goal",PoseStamped,queue_size=1,latch=True)
        self.timer_take_off=rospy.Timer(rospy.Duration(0.002), self.timerTakeOffCB)
        self.timer_take_off.shutdown()

        # self.alt_taken_off = 2.5; #Altitude when hovering after taking off
        self.alt_ground = 0; #Altitude of the ground
        self.initialized=False

        self.is_kill = False
        self.takeoff_goal=Goal()

        # Uncomment to test planning while stationary            
        # self.whoplans.value=self.whoplans.PANTHER
        # self.sendWhoPlans()

    #In rospy, the callbacks are all of them in separate threads
    def stateCB(self, data):
        self.pose.position.x = data.pos.x
        self.pose.position.y = data.pos.y
        self.pose.position.z = data.pos.z
        self.pose.orientation = data.quat

        if(self.initialized==False):
            #self.pubFirstTerminalGoal() Not needed
            self.initialized=True

    #Called when buttom pressed in the interface
    def globalflightmodeCB(self,req):
        if(self.initialized==False):
            print ("Not initialized yet. Is DRONE_NAME/state being published?")
            return

        if req.mode == req.GO and self.whoplans.value==self.whoplans.OTHER:
            print ("Starting taking off")
            self.takeOff()
            self.whoplans.value = self.whoplans.PANTHER
            # print("Taking off done")

        if req.mode == req.KILL:
            print ("Killing")
            self.kill()
            print ("Killed done")

        if req.mode == req.LAND and self.whoplans.value==self.whoplans.PANTHER:
            print ("Landing")
            self.land()
            print ("Landing done")

    def sendWhoPlans(self):
        self.whoplans.header.stamp = rospy.get_rostime()
        self.pubWhoPlans.publish(self.whoplans)


    def takeOff(self):
        print("In takeOff")
        self.is_kill = False
        self.takeoff_goal.p.x = self.pose.position.x;
        self.takeoff_goal.p.y = self.pose.position.y;
        self.takeoff_goal.p.z = self.pose.position.z;
        self.takeoff_goal.psi = quat2yaw(self.pose.orientation)
        self.takeoff_goal.power= True; #Turn on the motors
        self.timer_take_off=rospy.Timer(rospy.Duration(0.002), self.timerTakeOffCB)

    def timerTakeOffCB(self, event):
        alt_taken_off = 1.8; #Altitude when hovering after taking off
        self.takeoff_goal.p.z = min(self.takeoff_goal.p.z+0.0005, alt_taken_off);
        rospy.loginfo_throttle(0.5, "Taking off..., error={}".format(self.pose.position.z-alt_taken_off) )
        self.sendGoal(self.takeoff_goal)

        if abs(self.pose.position.z-alt_taken_off)<0.1:
            self.timer_take_off.shutdown()
            self.whoplans.value=self.whoplans.PANTHER
            self.sendWhoPlans()

    def land(self):
        self.is_kill = False
        self.whoplans.value=self.whoplans.OTHER
        self.sendWhoPlans();
        goal=Goal();
        goal.p.x = self.pose.position.x;
        goal.p.y = self.pose.position.y;
        goal.p.z = self.pose.position.z;
        goal.power= True; #Motors still on
        print ("self.pose.orientation= ", self.pose.orientation)
        goal.psi = quat2yaw(self.pose.orientation)
        print ("goal.psi= ", goal.psi)


        #Note that self.pose.position is being updated in the parallel callback
        while(abs(self.pose.position.z-self.alt_ground)>0.1):
            goal.p.z = max(goal.p.z-0.0035, self.alt_ground);
            rospy.sleep(0.01)
            self.sendGoal(goal)
        #Kill motors once we are on the ground
        self.kill()

    def kill(self):
        self.is_kill = True
        self.timer_take_off.shutdown()
        self.whoplans.value=self.whoplans.OTHER
        self.sendWhoPlans()
        goal=Goal();
        goal.p.x = self.pose.position.x;
        goal.p.y = self.pose.position.y;
        goal.p.z = self.pose.position.z;
        goal.psi = quat2yaw(self.pose.orientation)
        goal.power=False #Turn off the motors 
        self.sendGoal(goal) #TODO: due to race conditions, publishing whoplans.OTHER and then goal.power=False does NOT guarantee that the external planner doesn't publish a goal with power=true
                            #The easy workaround is to click several times in the 'kill' button of the GUI


    def sendGoal(self, goal):
        # goal.psi = quat2yaw(self.pose.orientation)
        goal.header.stamp = rospy.get_rostime()
        # print("[panther_cmds.py] Sending goal.yaw=",goal.psi);
        self.pubGoal.publish(goal)

    # def pubFirstTerminalGoal(self):
    #     msg=PoseStamped()
    #     msg.pose.position.x=self.pose.position.x
    #     msg.pose.position.y=self.pose.position.y
    #     msg.pose.position.z=1.0
    #     msg.pose.orientation = self.pose.orientation
    #     msg.header.frame_id="world"
    #     msg.header.stamp = rospy.get_rostime()
    #     self.pubClickedPoint.publish(msg)

                  
def startNode():
    c = Panther_Commands()
    #s = rospy.Service("/change_mode",MissionModeChange,c.srvCB)
    rospy.Subscriber("state", State, c.stateCB)
    rospy.Subscriber("/globalflightmode", QuadFlightMode, c.globalflightmodeCB)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('panther_commands')  
    startNode()
    print ("Behavior selector started") 