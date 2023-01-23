#!/usr/bin/env python

import numpy as np
import sympy as sp
import rospy
import rospkg
import collections
from datetime import date

def getTrefoil(tt,offset,slower,lim_x, lim_y, lim_z):
    x=(sp.sin((tt+slower*offset)/slower)+2*sp.sin(2*((tt+slower*offset)/slower))+3)/6; # in [0,1] approx
    y=(sp.cos((tt+slower*offset)/slower)-2*sp.cos(2*((tt+slower*offset)/slower))+3)/6;
    z=((-sp.sin(3*((tt+slower*offset)/slower)))+1.0)/2.0; # in [0,1] approx

    x=min(lim_x)+(max(lim_x)-min(lim_x))*x
    y=min(lim_y)+(max(lim_y)-min(lim_y))*y
    z=min(lim_z)+(max(lim_z)-min(lim_z))*z

    return [x, y, z]

if __name__=="__main__":

    # get package path
    package_path=rospkg.RosPack().get_path('panther')

    # Obstacle Params
    Obstacle = collections.namedtuple('Name', ["name","bbox", "slower", "offset", "lim_x", "lim_y", "lim_z"])
    all_drones=[        #"name"     ,"bbox"         ,"slower","offset","lim_x"    ,"lim_y"    ,"lim_z"
                Obstacle("obstacle1",[1.0, 1.0, 1.0],1.5     ,0.0     ,[-2.0, 2.0],[-2.0, 2.0],[1.0, 2.5])
                ]

    # symbolic t
    t=sp.symbols('t')

    # create obstacle yamls in the spedicified path
    for i in range(len(all_drones)):
        drone_i=all_drones[i];
        traj=getTrefoil(t, drone_i.offset, drone_i.slower, drone_i.lim_x, drone_i.lim_y, drone_i.lim_z)
        # print traj
        name_file=package_path+"/param/obstacle1.yaml"
        f = open(name_file, "w")
        f.write("# DO NOT EDIT. RUN THE PYTHON FILE INSTEAD TO GENERATE THIS .yaml FILE \n")
        f.write("# date: "+str(date.today())+"\n")
        f.write("traj_x: "+str(traj[0])+"\n")
        f.write("traj_y: "+str(traj[1])+"\n")
        f.write("traj_z: "+str(traj[2])+"\n")
        f.write("bbox: "+str(drone_i.bbox)+"\n")
        f.close()
        print ("Writing to " + name_file)