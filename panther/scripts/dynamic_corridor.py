#!/usr/bin/env python

# /* ----------------------------------------------------------------------------
#  * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  * -------------------------------------------------------------------------- */

import random
import roslib
import rospy
import math
from panther_msgs.msg import DynTraj
from snapstack_msgs.msg import Goal, State
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np
from numpy import linalg as LA
import random
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf
from math import sin, cos, tan, floor
from numpy import sign as sgn #Because https://github.com/ArashPartow/exprtk  uses sgn, not sign
import os
import copy 
import sys
import glob
import rospkg
import argparse
from datetime import datetime
import yaml

def getColorJet(v, vmin, vmax): 

  c=ColorRGBA()

  c.r = 1
  c.g = 1
  c.b = 1
  c.a = 1

  if (v < vmin):
    v = vmin
  if (v > vmax):
    v = vmax
  dv = vmax - vmin

  if (v < (vmin + 0.25 * dv)):
    c.r = 0
    c.g = 4 * (v - vmin) / dv
  
  elif (v < (vmin + 0.5 * dv)):
    c.r = 0
    c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv
  
  elif (v < (vmin + 0.75 * dv)):
    c.r = 4 * (v - vmin - 0.5 * dv) / dv
    c.b = 0
  
  else:
    c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv
    c.b = 0

  return c

class DynCorridor:

    def getTrajectoryPosMeshBBox(self, i):
        delta_beginning=2.0

        delta=(self.x_max-self.x_min-delta_beginning)/(self.total_num_obs)
        x=delta_beginning + self.x_min + i*delta #random.uniform(self.x_min, self.x_max);
        y=random.uniform(self.y_min, self.y_max)
        z=random.uniform(self.z_min, self.z_max)
        offset=random.uniform(-2*math.pi, 2*math.pi)

        # x = 3
        # y = 0
        # z = 1
        # offset = 0

        slower=random.uniform(self.slower_min, self.slower_max)
        s=self.scale
        if(self.getType(i)=="dynamic"):
            mesh=random.choice(self.available_meshes_dynamic)
            bbox=self.bbox_dynamic; 
            if(self.type_of_obst_traj=="trefoil"):
                [x_string, y_string, z_string] = self.trefoil(x,y,z, self.scale[0],self.scale[1],self.scale[2], offset, slower)
            elif(self.type_of_obst_traj=="eightCurve"):
                [x_string, y_string, z_string] = self.eightCurve(x,y,z, self.scale[0],self.scale[1],self.scale[2], offset, slower)
            elif(self.type_of_obst_traj=="square"):
                [x_string, y_string, z_string] = self.square(x,y,z, self.scale[0],self.scale[1],self.scale[2], offset, slower)
            elif(self.type_of_obst_traj=="epitrochoid"):
                [x_string, y_string, z_string] = self.epitrochoid(x,y,z, self.scale[0],self.scale[1],self.scale[2], offset, slower)
            elif(self.type_of_obst_traj=="static"):
                [x_string, y_string, z_string] = self.static(2.5,0.0,1.0)    
            else:
                print("*******  TRAJECTORY ["+ self.type_of_obst_traj+"] "+" NOT SUPPORTED   ***********")
                exit();         

        else:
            mesh=random.choice(self.available_meshes_static)
            bbox=self.bbox_static_vert
            z=bbox[2]/2.0
            [x_string, y_string, z_string] = self.wave_in_z(x, y, z, self.scale[2], offset, 1.0)
        return [x_string, y_string, z_string, x, y, z, mesh, bbox]

    def getType(self,i):
        if(i<self.num_of_dyn_objects):
            return "dynamic"
        else:
            return "static"

    def __init__(self, total_num_obs,gazebo, type_of_obst_traj, alpha_scale_obst_traj, beta_faster_obst_traj):

        random.seed(datetime.now())

        self.state=State()

        name = rospy.get_namespace()
        self.name = name[1:-1]

        self.total_num_obs=total_num_obs
        self.num_of_dyn_objects=int(1.0*total_num_obs)
        self.num_of_stat_objects=total_num_obs-self.num_of_dyn_objects; 
        self.x_min= 1.0 
        self.x_max= 3.0
        self.y_min= 1.0 
        self.y_max= 3.0
        self.z_min= 1.0
        self.z_max= 1.0
        # self.scale= [(self.x_max-self.x_min)/self.total_num_obs, 5.0, 1.0]
        self.scale= [alpha_scale_obst_traj, alpha_scale_obst_traj, alpha_scale_obst_traj]
        self.slower_min=3.0   #1.2 or 2.3
        self.slower_max=3.0   #1.2 or 2.3

        PANTHER_YAML_PATH = rospkg.RosPack().get_path("panther") + "/param/panther.yaml"
        with open(PANTHER_YAML_PATH) as f:
            PANTHER_YAML_PARAMS = yaml.safe_load(f)

        self.bbox_dynamic=PANTHER_YAML_PARAMS["obstacle_bbox"] # this corresponds to training_obst_size defined in panther.yaml
        self.add_noise_to_obst = PANTHER_YAML_PARAMS["add_noise_to_obst"]
        self.bbox_static_vert=[0.4, 0.4, 4]
        self.bbox_static_horiz=[0.4, 8, 0.4]
        self.percentage_vert=0.0
        self.name_obs="obs_"
        self.max_vel_obstacles=-10.0

        self.type_of_obst_traj=type_of_obst_traj #eightCurve, static, square, epitrochoid

        self.available_meshes_static=["package://panther/meshes/ConcreteDamage01b/model3.dae", "package://panther/meshes/ConcreteDamage01b/model2.dae"]
        self.available_meshes_dynamic=["package://panther/meshes/ConcreteDamage01b/model4.dae"]

        self.marker_array=MarkerArray()
        self.all_dyn_traj=[]
        self.all_dyn_traj_zhejiang=[]

        self.total_num_obs=self.num_of_dyn_objects + self.num_of_stat_objects

        for i in range(self.total_num_obs): 
            [traj_x, traj_y, traj_z, x, y, z, mesh, bbox]=self.getTrajectoryPosMeshBBox(i);           
            self.marker_array.markers.append(self.generateMarker(mesh, bbox, i))

            dynamic_trajectory_msg=DynTraj(); 
            dynamic_trajectory_msg.use_pwp_field=False
            dynamic_trajectory_msg.is_agent=False
            dynamic_trajectory_msg.header.stamp= rospy.Time.now()
            dynamic_trajectory_msg.s_mean = [traj_x, traj_y, traj_z]
            dynamic_trajectory_msg.s_var = ["0.001", "0.001", "0.001"] #TODO (a nonzero variance is needed to choose the obstacle to focus on, see panther.cpp)
            dynamic_trajectory_msg.bbox = [bbox[0], bbox[1], bbox[2]]
            dynamic_trajectory_msg.pos.x=x #Current position, will be updated later
            dynamic_trajectory_msg.pos.y=y #Current position, will be updated later
            dynamic_trajectory_msg.pos.z=z #Current position, will be updated later
            dynamic_trajectory_msg.id = 4000 + i #Current id 4000 to avoid interference with ids from agents #TODO
            dynamic_trajectory_msg.is_committed = True

            self.all_dyn_traj.append(dynamic_trajectory_msg)

        self.all_dyn_traj_zhejiang=copy.deepcopy(self.all_dyn_traj)

        self.pubTraj = rospy.Publisher('/trajs', DynTraj, queue_size=1, latch=True)
        self.pubShapes_dynamic_mesh = rospy.Publisher('/obstacles_mesh', MarkerArray, queue_size=1, latch=True)

        # self.pubShapes_dynamic_mesh_zhejiang = rospy.Publisher('/obstacles_mesh_zhejiang', MarkerArray, queue_size=1, latch=True)
        self.pubShapes_dynamic_mesh_colored = rospy.Publisher('/obstacles_mesh_colored', MarkerArray, queue_size=1, latch=True)
        self.pubTraj_zhejiang = rospy.Publisher('/SQ01s/trajs_zhejiang', DynTraj, queue_size=1, latch=True)

        #self.pubGazeboState = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=100)

        if(gazebo):
            # Spawn all the objects in Gazebo
            for i in range(self.total_num_obs):
                self.spawnGazeboObstacle(i)

        rospy.sleep(0.5)

    def generateMarker(self, mesh, bbox, i):
        marker=Marker()
        marker.id=i
        marker.ns="mesh"
        marker.header.frame_id="world"
        marker.type=marker.MESH_RESOURCE
        marker.action=marker.ADD

        marker.pose.position.x=0.0 #Will be updated later
        marker.pose.position.y=0.0 #Will be updated later
        marker.pose.position.z=0.0 #Will be updated later
        marker.pose.orientation.x=0.0
        marker.pose.orientation.y=0.0
        marker.pose.orientation.z=0.0
        marker.pose.orientation.w=1.0
        marker.lifetime = rospy.Duration.from_sec(0.0)
        marker.mesh_use_embedded_materials=True
        marker.mesh_resource=mesh

        marker.scale.x=bbox[0]
        marker.scale.y=bbox[1]
        marker.scale.z=bbox[2]
        return marker

    def pubTF(self, timer):
        br = tf.TransformBroadcaster()

        marker_array_static_mesh=MarkerArray()
        marker_array_dynamic_mesh=MarkerArray()

        for i in range(self.total_num_obs): 
            t_ros=rospy.Time.now()
            t=rospy.get_time(); #Same as before, but it's float

            marker=self.marker_array.markers[i]

            x = eval(self.all_dyn_traj[i].s_mean[0])
            y = eval(self.all_dyn_traj[i].s_mean[1])
            z = eval(self.all_dyn_traj[i].s_mean[2])

            # Set the stamp and the current pos
            self.all_dyn_traj[i].header.stamp= t_ros
            self.all_dyn_traj[i].pos.x=x #Current position
            self.all_dyn_traj[i].pos.y=y #Current position
            self.all_dyn_traj[i].pos.z=z #Current position

            self.pubTraj.publish(self.all_dyn_traj[i])
            br.sendTransform((x, y, z), (0,0,0,1), t_ros, self.name_obs+str(self.all_dyn_traj[i].id), "world")

            self.marker_array.markers[i].pose.position.x=x
            self.marker_array.markers[i].pose.position.y=y
            self.marker_array.markers[i].pose.position.z=z

            #If you want to see the objects in rviz

        self.pubShapes_dynamic_mesh.publish(self.marker_array)

    def static(self,x,y,z):
        return [str(x), str(y), str(z)]

    # Trefoil knot, https://en.wikipedia.org/wiki/Trefoil_knot
    def trefoil(self,x,y,z,scale_x, scale_y, scale_z, offset, slower):

        #slower=1.0; #The higher, the slower the obstacles move" 
        tt='t/' + str(slower)+'+'

        x_string=str(scale_x/6.0)+'*(sin('+tt +str(offset)+') + 2 * sin(2 * '+tt +str(offset)+'))' +'+' + str(x); #'2*sin(t)' 
        y_string=str(scale_y/5.0)+'*(cos('+tt +str(offset)+') - 2 * cos(2 * '+tt +str(offset)+'))' +'+' + str(y); #'2*cos(t)' 
        z_string=str(scale_z/2.0)+'*(-sin(3 * '+tt +str(offset)+'))' + '+' + str(z);                              #'1.0'

        return [x_string, y_string, z_string]

    def square(self,x,y,z,scale_x, scale_y, scale_z, offset, slower):

        #slower=1.0; #The higher, the slower the obstacles move" 
        tt='t/' + str(slower)+'+'

        tt='(t+'+str(offset)+')'+'/' + str(slower)
        cost='cos('+tt+')'
        sint='sin('+tt+')'

        #See https://math.stackexchange.com/questions/69099/equation-of-a-rectangle
        x_string=str(scale_x)+'*0.5*(abs('+cost+')*'+cost + '+abs('+sint+')*'+sint+')'
        y_string=str(scale_x)+'*0.5*(abs('+cost+')*'+cost + '-abs('+sint+')*'+sint+')'
        z_string=x_string

        #Without rotation
        x_string=x_string+'+' + str(x)
        y_string=y_string+'+' + str(y)
        z_string=z_string+'+' + str(z)

        return [x_string, y_string, z_string]

    def wave_in_z(self,x,y,z,scale, offset, slower):

        tt='t/' + str(slower)+'+'

        x_string=str(x)
        y_string=str(y)
        z_string=str(scale)+'*(-sin( '+tt +str(offset)+'))' + '+' + str(z)               

        return [x_string, y_string, z_string]

    def eightCurve(self,x,y,z,scale_x, scale_y, scale_z,offset,slower): #The xy projection is an https://mathworld.wolfram.com/v.html

        tt='(t+'+str(offset)+')'+'/' + str(slower)
        cost='cos('+tt+')'
        sint='sin('+tt+')'

        x_string=str(scale_x)+'*'+sint + '+' + str(x)
        y_string=str(scale_y)+'*'+sint+'*'+cost + '+' + str(y)
        z_string=str(scale_z)+'*'+sint+'*'+sint+'*'+cost + '+' + str(z)

        return [x_string, y_string, z_string]

    
    def epitrochoid(self,x,y,z,scale_x, scale_y, scale_z, offset, slower):

        #slower=1.0; #The higher, the slower the obstacles move" 
        tt='(t+'+str(offset)+')'+'/' + str(slower)
        cost='cos('+tt+')'
        sint='sin('+tt+')'

        a=1.0
        b=0.4
        h=4.0

        aplusb='('+str(a)+'+'+str(b)+')'
        adivbplus1='('+str(a)+'/'+str(b)+'+1)'

        x_string= str(scale_x/7)+'*'+'(' + aplusb + '*' + cost + '-' + str(h) + '*cos(' + adivbplus1 + '*'+tt +'))+' + str(x)    #str(scale_x/6.0)+'*(sin('+tt +str(offset)+') + 2 * sin(2 * '+tt +str(offset)+'))' +'+' + str(x); #'2*sin(t)' 
        y_string= str(scale_y/7)+'*'+'(' + aplusb + '*' + sint + '-' + str(h) + '*sin(' + adivbplus1 + '*'+tt +'))+' + str(y)
        z_string= str(scale_z)+'*'+'(' +  sint + '*' + cost  +')+' + str(z)

        return [x_string, y_string, z_string]

    def spawnGazeboObstacle(self, i):

            rospack = rospkg.RosPack()
            path_panther=rospack.get_path('panther')
            path_file=path_panther+"/meshes/tmp_"+str(i)+".urdf"

            f = open(path_file, "w") #TODO: This works, but it'd better not having to create this file
            scale=self.marker_array.markers[i].scale
            scale='"'+str(scale.x)+" "+str(scale.y)+" "+str(scale.z)+'"'

            x=self.all_dyn_traj[i].pos.x
            y=self.all_dyn_traj[i].pos.y
            z=self.all_dyn_traj[i].pos.z

            #Remember NOT to include de <collision> tag (Gazebo goes much slower if you do so)
            f.write("""
<robot name="name_robot">
  <link name="name_link">
    <inertial>
      <mass value="0.200" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="5.8083e-4" ixy="0" ixz="0" iyy="3.0833e-5" iyz="0" izz="5.9083e-4" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="""+'"'+self.marker_array.markers[i].mesh_resource+'"'+""" scale="""+scale+"""/>
      </geometry>
    </visual>
  </link>
  <gazebo>
    <plugin name="move_model" filename="libmove_model.so">
    <traj_x>"""+self.all_dyn_traj[i].s_mean[0]+"""</traj_x>
    <traj_y>"""+self.all_dyn_traj[i].s_mean[1]+"""</traj_y>
    <traj_z>"""+self.all_dyn_traj[i].s_mean[2]+"""</traj_z>
    </plugin>
  </gazebo>
</robot>
                """)
  # <plugin name="pr2_pose_test" filename="libpr2_pose_test.so"/>
            f.close()

            # os.system("rosrun gazebo_ros spawn_model -file `rospack find panther`/meshes/tmp.urdf -urdf -x " + str(x) + " -y " + str(y) + " -z " + str(z) + " -model "+self.name_obs+str(i)+" &"); #all_
            os.system("rosrun gazebo_ros spawn_model -file "+path_file+" -urdf -x " + str(x) + " -y " + str(y) + " -z " + str(z) + " -model "+self.name_obs+str(i)+" && rm "+path_file + " &"); #all_
            # os.remove(path_file)

             
#https://stackoverflow.com/a/43357954/6057617
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--num_of_obs', type=int, required=True)
    parser.add_argument('--gazebo', type=str2bool, required=True)
    parser.add_argument('--type_of_obst_traj', type=str, required=True)
    parser.add_argument('--alpha_scale_obst_traj', type=float, required=True)
    parser.add_argument('--beta_faster_obst_traj', type=float, required=True)
    # Parse the argument
    print(sys.argv)
    print("*************************************")
    args = parser.parse_args(sys.argv[1:11]) #See https://discourse.ros.org/t/getting-python-argparse-to-work-with-a-launch-file-or-python-node/10606

    ns = rospy.get_namespace()
    try:
        rospy.init_node('dynamic_obstacles')
        c = DynCorridor(args.num_of_obs, args.gazebo, args.type_of_obst_traj, args.alpha_scale_obst_traj, args.beta_faster_obst_traj)
        rospy.Timer(rospy.Duration(0.01), c.pubTF)
        c.pubTF(None)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
