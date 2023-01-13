FROM ros:noetic 

LABEL Maintainer="Kota Kondo <kkondo@mit.edu>" \
      Description="Ubuntu 20.04, ROS Noetic for Deep-PANTHER"

# Usuals
##########################################
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
      apt-get upgrade -y && \
      apt-get install -q -y --no-install-recommends git tmux vim apt-utils && \
      apt-get install -q -y --no-install-recommends wget && \
      apt-get autoremove -y && \
      apt-get clean

# IPOPT
##########################################
RUN apt-get install -y gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends && \
      apt-get install -y coinor-libipopt1v5 coinor-libipopt-dev

# CasADi
##########################################
RUN apt-get remove swig swig3.0 swig4.0 && \
      mkdir -p /home/kota/installations
WORKDIR /home/kota/installations
RUN git clone https://github.com/jaeandersson/swig
WORKDIR /home/kota/installations/swig
RUN git checkout -b matlab-customdoc origin/matlab-customdoc && \
      sh autogen.sh && \
      apt-get install gcc-7 g++-7 bison byacc -y && \
      ./configure CXX=g++-7 CC=gcc-7 && \
      make && make install

WORKDIR /home/kota/installations
RUN mkdir casadi
WORKDIR /home/kota/installations/casadi
RUN git clone https://github.com/casadi/casadi
RUN mkdir build
WORKDIR /home/kota/installations/casadi/casadi/build
#You may need to run the command twice until the output says that `Ipopt` has been detected (although `IPOPT` is also being detected when you run it for the first time)
RUN cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON .. && \
      cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON .. && \
      make -j20 && make install

# Create a virtual python env
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
##########################################
# ENV VIRTUAL_ENV=/home/kota/installations/venvs_python/my_venv
# RUN apt-get install python3-venv -y
# RUN mkdir -p /home/kota/installations/venvs_python/my_venv
# RUN python3 -m venv $VIRTUAL_ENV
# RUN PATH="$VIRTUAL_ENV/bin:$PATH"
# RUN printf '\nalias activate_my_venv="source /home/kota/installations/venvs_python/my_venv/bin/activate"' >> ~/.bashrc 
# RUN . /home/kota/installations/venvs_python/my_venv/bin/activate

# Deep PANTHER
##########################################
RUN apt-get install git-lfs ccache 
WORKDIR /home/kota/
RUN mkdir ws
WORKDIR /home/kota/ws
RUN mkdir src
WORKDIR /home/kota/ws/src
RUN git clone --recursive https://github.com/mit-acl/deep_panther.git 
WORKDIR /home/kota/ws/src/deep_panther
RUN git lfs install
WORKDIR /home/kota/ws/src/deep_panther/panther_compression/imitation
RUN apt-get install python3-pip -y && \
      pip3 install numpy Cython wheel seals rospkg defusedxml empy pyquaternion pytest && \
      pip3 install -e . # pip3 install -e . is editable mode (looks for setup.py)
RUN apt-get install python3-catkin-tools -y && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y ros-"${ROS_DISTRO}"-rviz-visual-tools ros-"${ROS_DISTRO}"-pybind11-catkin ros-"${ROS_DISTRO}"-tf2-sensor-msgs ros-"${ROS_DISTRO}"-jsk-rviz-plugins && \
      apt-get install --reinstall python3-rospkg python3-rosdep python3-catkin-pkg python3-rosinstall python3-vcstools && \
      apt update && apt install ros-noetic-desktop-full -y
WORKDIR /home/kota/ws/
RUN catkin config --extend /opt/ros/noetic && \
      catkin build  && \
      printf '\nsource /home/kota/ws/devel/setup.bash' >> ~/.bashrc #Remember to change PATH_TO_YOUR_WS  && \
      printf '\nexport PYTHONPATH="${PYTHONPATH}:$(rospack find panther)/../panther_compression"' >> ~/.bashrc   && \
      . ~/.bashrc

# Linear Solver (if you don't run expert, then you don't need this)
##########################################
# RUN apt-get install libblas3gf libblas-dev liblapack3gf liblapack-dev gfortran
# 1. Download the coinhsl package from http://www.hsl.rl.ac.uk/ipopt/
# 2. Place it 
# 2. This code will copy it into Docker and compile

WORKDIR /home/kota/installations
ADD coinhsl-archive-2021.05.05 /home/kota/installations/coinhsl-archive-2021.05.05
WORKDIR /home/kota/installations/coinhsl-archive-2021.05.05
#sudo make uninstall && sudo make clean #Only needed if you have installed it before
RUN ./configure LIBS="-llapack" --with-blas="-L/usr/lib -lblas" CXXFLAGS="-g -O3 -fopenmp" FCFLAGS="-g -O3 -fopenmp" CFLAGS="-g -O3 -fopenmp" #the output should say `checking for metis to compile... yes`
RUN make install #(the files will go to /usr/local/lib)
WORKDIR /usr/local/lib
RUN ln -s libcoinhsl.so libhsl.so #(This creates a symbolic link `libhsl.so` pointing to `libcoinhsl.so`). See https://github.com/casadi/casadi/issues/1437
RUN printf '\nexport LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"' >> ~/.bashrc

RUN pip3 install numpy==1.21.1
RUN pip3 install python-dateutil==2.8.2
RUN pip3 install imitation
RUN pip3 install matplotlib
RUN pip3 install numpy==1.25.0

# Useful things in ~/.bashrc
##########################################
RUN printf '\nsource /opt/ros/noetic/setup.bash' >> ~/.bashrc && \
      printf '\nalias cb="catkin build"' >> ~/.bashrc && \
      printf '\nalias gs="git status"' >> ~/.bashrc && \
      printf '\nalias gp="git push"' >> ~/.bashrc && \
      printf '\nalias kr="killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f panther & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill panther_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f panther_commands & pkill -f dynamic_corridor & tmux kill-server & pkill -f perfect_controller & pkill -f publish_in_gazebo"' >> ~/.bashrc && \
      printf '\nalias py="python"' >> ~/.bashrc && \
      printf '\nalias tks="tmux kill-server"' >> ~/.bashrc

WORKDIR /home/kota/ws/src/deep_panther