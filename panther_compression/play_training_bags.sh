killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill mader_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f mader_commands & pkill -f dynamic_corridor & tmux kill-server 
rosbag reindex *.bag
rm training*.*.bag #Remove the original bags (i.e., keep only the indexed ones)
python concatenate_bags.py "training*.bag"
roscore &
sleep 0.5
rosrun rviz rviz -d "./rviz_cfgs/rviz_compression.rviz" &
rosbag play training_concatenated.bag -r 0.1