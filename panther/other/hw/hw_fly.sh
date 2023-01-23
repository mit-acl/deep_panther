#!/bin/bash

# this script creates multiple windows for deep panther

# session and window name
SESSION=panther
WINDOW=base_station

# creates the session with a name and renames the window name
cmd="new-session -d -s $SESSION -x- -y-; rename-window $WINDOW"
tmux -2 $cmd

# window number
w=0

# split tmux into 2x2
################ 
for i in {1..3}
do
	tmux split-window -h
	tmux select-layout -t $SESSION:$w.$i even-horizontal
	tmux select-pane -t $SESSION:$w.$i
done

for i in 0 2 4
do 
	tmux select-pane -t $SESSION:$w.$i
	tmux split-window -v
done

for i in 1
do
	tmux resize-pane -t $SESSION:$w.$i -y 20
done
################

# wait for .bashrc to load
sleep 1

# ssh into voxl
tmux send-keys -t $SESSION:$w.0 "ssh root@nx04.local" C-m

# ssh into nuc
tmux send-keys -t $SESSION:$w.1 "ssh nuc4@192.168.0.103" C-m
tmux send-keys -t $SESSION:$w.2 "ssh nuc4@192.168.0.103" C-m

sleep 3

# ntp date
tmux send-keys -t $SESSION:$w.1 "sudo ntpdate time.nist.gov" C-m
sleep 10


tmux send-keys -t $SESSION:$w.1 "roslaunch panther hw_onboard quad:=NX04" C-m #by using /home/nuc1/ instead of ~/, we can stop record data on sikorsky when we are not using the vehicle.

sleep 3
tmux send-keys -t $SESSION:$w.0 "fly" C-m

# tf file
tmux send-keys -t $SESSION:$w.2 "python tf_timestamp.py" C-m

# base station
tmux send-keys -t $SESSION:$w.3 "roslaunch panther hw_base_station.launch" C-m

# obstacle fly
tmux send-keys -t $SESSION:$w.4 "ssh root@nx10.local" C-m
sleep 3
tmux send-keys -t $SESSION:$w.4 "fly" C-m

tmux send-keys -t $SESSION:$w.5 "roslaunch panther hw_obstacle.launch obs1:=NX10" C-m



tmux -2 attach-session -t $SESSION