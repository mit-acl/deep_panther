FROM ros:noetic

LABEL Maintainer="Kota Kondo <kkondo@mit.edu>" \
      Description="Ubuntu 20.04, ROS Noetic for Robust MADER"

RUN sudo rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
      apt-get upgrade -y && \
      apt-get install -q -y --no-install-recommends git tmux vim apt-utils && \
      apt-get install -q -y --no-install-recommends wget && \
      apt-get autoremove -y && \
      apt-get clean

#IPOPT
RUN apt-get install -y gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends
RUN apt-get install -y coinor-libipopt1v5 coinor-libipopt-dev

# CasADi
##########################################
RUN apt-get remove swig swig3.0 swig4.0 
RUN mkdir -p /home/kota/installations
WORKDIR /home/kota/installations
RUN git clone https://github.com/jaeandersson/swig
WORKDIR /home/kota/installations/swig
RUN git checkout -b matlab-customdoc origin/matlab-customdoc        
RUN sh autogen.sh
RUN apt-get install gcc-7 g++-7 bison byacc -y
RUN ./configure CXX=g++-7 CC=gcc-7            
RUN make
RUN sudo make install

WORKDIR /home/kota/installations
RUN mkdir casadi
WORKDIR /home/kota/installations/casadi
RUN git clone https://github.com/casadi/casadi
WORKDIR /home/kota/installations/casadi/casadi 
RUN mkdir build
WORKDIR /home/kota/installations/casadi/casadi/build
RUN cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON ..
RUN cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON ..
#You may need to run the command above twice until the output says that `Ipopt` has been detected (although `IPOPT` is also being detected when you run it for the first time)
RUN make -j20
RUN make install

# Create a virtual python env
RUN apt-get install python3-venv -y
WORKDIR /home/kota/installations
RUN mkdir venvs_python
WORKDIR /home/kota/installations/venvs_python 
RUN python3 -m venv ./my_venv
RUN printf '\nalias activate_my_venv="source ~/installations/venvs_python/my_venv/bin/activate"' >> ~/.bashrc
RUN . /home/kota/installations/venvs_python/my_venv/bin/activate

# Deep PANTHER
RUN apt-get install git-lfs ccache 
WORKDIR /home/kota/
RUN mkdir ws
WORKDIR /home/kota/ws
RUN mkdir src
WORKDIR /home/kota/ws/src
RUN git clone https://github.com/mit-acl/deep_panther
WORKDIR /home/kota/ws/src/deep_panther
RUN git lfs install
RUN git submodule init && git submodule update
WORKDIR /home/kota/ws/src/deep_panter/panther_compression/imitation
RUN apt-get install python3-pip -y
RUN pip install numpy Cython wheel seals rospkg defusedxml empy pyquaternion pytest
# RUN pip install -e .
RUN apt-get install python3-catkin-tools -y
# RUN apt-get install -y ros-"${ROS_DISTRO}"-rviz-visual-tools ros-"${ROS_DISTRO}"-pybind11-catkin ros-"${ROS_DISTRO}"-tf2-sensor-msgs ros-"${ROS_DISTRO}"-jsk-rviz-plugins

#INSTALL python-catkin-tools (to be able to use catkin build)
##########################################
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'
RUN wget http://packages.ros.org/ros.key -O - |  apt-key add -
RUN apt-get update && apt-get install python3-catkin-tools -y

WORKDIR /home/kota/ws/
RUN catkin build
RUN printf '\nsource PATH_TO_YOUR_WS/devel/setup.bash' >> ~/.bashrc #Remember to change PATH_TO_YOUR_WS
RUN printf '\nexport PYTHONPATH="${PYTHONPATH}:$(rospack find panther)/../panther_compression"' >> ~/.bashrc 
RUN . ~/.bashrc


# RUN echo source /opt/ros/noetic/setup.bash >> ~/.bashrc

#CLONE SUBMODULES, INSTALL DEPENDENCIES AND COMPILE
##########################################
# RUN apt update
# RUN DEBIAN_FRONTEND=noninteractive rosdep install --from-paths /home/kota/rmader_ws/src --ignore-src -r -y
# RUN apt-get update && apt-get install libncurses5-dev -y
# RUN catkin config -DCMAKE_BUILD_TYPE=Release
# RUN echo source /home/kota/ws/devel/setup.bash >> ~/.bashrc

RUN echo 'alias cb="catkin build"' >> ~/.bashrc && \
      echo 'alias gs="git status"' >> ~/.bashrc && \
      echo 'alias gp="git push"' >> ~/.bashrc && \
      echo 'alias kr="killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f panther & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill rmader_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f rmader_commands & pkill -f dynamic_corridor & tmux kill-server & pkill -f perfect_controller & pkill -f publish_in_gazebo"' >> ~/.bashrc && \
      echo 'alias py="python"' >> ~/.bashrc && \
      echo 'alias rvmd="roscd rmader && cd rviz_cfgs && rosrun rviz rviz -d rmader.rviz"' >> ~/.bashrc && \
      echo 'alias tks="tmux kill-server"' >> ~/.bashrc && \
      apt-get install psmisc && \
      apt-get update && \
      pip3 install -U bagpy
      # . ~/.bashrc && \
      # roscd rmader && cd scripts && \

# COPY to-be-copied/my-entrypoint.bash ~
# ENTRYPOINT ["my-entrypoint.bash"]