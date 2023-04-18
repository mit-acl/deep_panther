#Partly taken from http://library.isr.ist.utl.pt/docs/roswiki/rosbag(2f)Cookbook.html

#In a rosbag, the messages are played (or shown in rqt_bag) using the time when they where received (not the time of the msgs' header timestamps)
#This file create a new rosbag where the messages are written according to their header timestamps

import rosbag
import sys
import glob
import os
import rospy
import os.path
# from visualization_msgs.msg import MarkerArray


if (len(sys.argv) <=1):
    print ("Usage is one of these: | python this_file.py name_of_bag.bag ")
    print ('                       | python this_file.py "2020_*.bag" (do not forget the "")')
    sys.exit(1)

new_name_bag="training_concatenated.bag"
os.system("rm "+new_name_bag)

name_bags=glob.glob(sys.argv[1])

if(len(name_bags)==0):
    print("No bag found with that name")

last_time=0.0

for name_bag in name_bags:
    # new_name_bag = name_bag.replace(".bag", "")+"_rewritten.bag"
    # raw_name_bag=os.path.splitext(name_bag)[0] 
    print("Writting ", name_bag, " --> ", new_name_bag)

    if(os.path.exists(new_name_bag)):
      option='a'
    else:
      option='w'

    offset_with_previous_bag=rospy.Duration.from_sec(last_time + 0.1)

    with rosbag.Bag(new_name_bag, option) as outbag:
        for topic, msg, t in rosbag.Bag(name_bag).read_messages():
            # This also replaces tf timestamps under the assumption 
            # that all transforms in the message share the same timestamp

            t_corrected=offset_with_previous_bag+t
            if topic == "/tf" and msg.transforms:
                for i in range(len(msg.transforms)):
                    msg.transforms[i].header.stamp=t_corrected
                outbag.write(topic, msg, t_corrected) #msg.transforms[0].header.stamp
            elif topic.startswith("/obs") : #MarkerArray
                for i in range(len(msg.markers)):
                    msg.markers[i].header.stamp=t_corrected
                outbag.write(topic, msg, t_corrected)
            else:
                msg.header.stamp=t_corrected
                outbag.write(topic, msg, t_corrected)

            last_time=(t_corrected).to_sec()
