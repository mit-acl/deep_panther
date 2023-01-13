FROM osrf/ros:noetic-desktop-full
SHELL ["/bin/bash", "-c"]

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

# Making directories and setting up the environment
##########################################
WORKDIR /home
RUN mkdir installations
RUN mkdir code

RUN export DEBIAN_FRONTEND=noninteractive

# Linear Solver (if you don't run expert, then you don't need this)
##########################################
# 1. Download the coinhsl package from http://www.hsl.rl.ac.uk/ipopt/
# 2. Download METIS from http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz
# 3. Place both zipped files in the same directory as this Dockerfile

# Install BLAS, LAPACK, and Fortran
RUN apt-get install -y libblas3 libblas-dev liblapack3 liblapack-dev gfortran

WORKDIR /home/installations

# Unzip coinhsl
ADD coinhsl-archive-2021.05.05.tar.gz /home/installations

WORKDIR /home/installations/coinhsl-archive-2021.05.05

# Unzip metis
ADD metis-4.0.3.tar.gz /home/installations/coinhsl-archive-2021.05.05

# Build and install and coinhsl
RUN ./configure LIBS="-llapack" --with-blas="-L/usr/lib -lblas" CXXFLAGS="-g -O3 -fopenmp" FCFLAGS="-g -O3 -fopenmp" CFLAGS="-g -O3 -fopenmp"
RUN make install

WORKDIR /usr/local/lib

# Link libcoinhsl.so to libhsl.so and export the path
RUN ln -s libcoinhsl.so libhsl.so #(This creates a symbolic link `libhsl.so` pointing to `libcoinhsl.so`). See https://github.com/casadi/casadi/issues/1437
RUN printf '\nexport LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"' >> ~/.bashrc

RUN source ~/.bashrc

# IPOPT
##########################################
RUN apt-get install -y gcc g++ git cmake pkg-config --install-recommends && \
    apt-get install -y coinor-libipopt1v5 coinor-libipopt-dev

# CasADi
##########################################
RUN apt-get remove swig swig3.0 swig4.0 && \
      mkdir -p /home/installations

WORKDIR /home/installations

# Clone and Install SWIG
RUN git clone https://github.com/jaeandersson/swig
WORKDIR /home/installations/swig
RUN git checkout -b matlab-customdoc origin/matlab-customdoc && \
      sh autogen.sh && \
      apt-get install -y gcc-7 g++-7 bison byacc && \
      ./configure CXX=g++-7 CC=gcc-7 && \
      make && make install

WORKDIR /home/installations

# Clone and Install CasADi
RUN mkdir casadi
WORKDIR /home/installations/casadi
RUN git clone https://github.com/casadi/casadi
WORKDIR /home/installations/casadi/casadi 
RUN mkdir build
WORKDIR /home/installations/casadi/casadi/build
# You may need to run the command twice until the output says that `Ipopt` has been detected (although `IPOPT` is also being detected when you run it for the first time)
RUN cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON .. && \
    cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON .. && \
    make -j20 && make install # -j20 is the number of cores you have

# # Python 3.8
# ##########################################
# RUN apt-get install py

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
WORKDIR /home/code/
RUN mkdir deep_panther

WORKDIR /home/code/deep_panther
RUN mkdir ws

WORKDIR /home/code/deep_panther/ws
RUN mkdir src

WORKDIR /home/code/deep_panther/ws/src
RUN git clone --recursive https://github.com/mit-acl/deep_panther.git 

WORKDIR /home/code/deep_panther/ws/src/deep_panther

# Install git lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN apt-get install -y git-lfs

RUN git lfs install

WORKDIR /home/code/deep_panther/ws/src/deep_panther/panther_compression/imitation
RUN apt-get install -y python3-pip && \
    pip3 install --upgrade numpy python-dateutil && \
    pip3 install numpy Cython wheel seals rospkg defusedxml empy pyquaternion pytest && \
    pip3 install -e . # pip3 install -e . is editable mode (looks for setup.py)

RUN apt-get install python3-catkin-tools -y && \
    apt-get install -y ros-"${ROS_DISTRO}"-rviz-visual-tools ros-"${ROS_DISTRO}"-pybind11-catkin ros-"${ROS_DISTRO}"-tf2-sensor-msgs ros-"${ROS_DISTRO}"-jsk-rviz-plugins && \
#     apt-get install -y --reinstall python3-rospkg python3-rosdep python3-catkin-pkg python3-rosinstall python3-vcstools && \
#     apt update && apt install -y ros-noetic-desktop-full

WORKDIR /home/code/deep_panther/ws/
RUN catkin config --extend /opt/ros/noetic && \
    catkin build  && \
    printf '\nsource /home/code/deep_panther/ws/devel/setup.bash' >> ~/.bashrc #Remember to change PATH_TO_YOUR_WS  && \
    printf '\nexport PYTHONPATH="${PYTHONPATH}:$(rospack find panther)/../panther_compression"' >> ~/.bashrc   && \
    source ~/.bashrc

# # Linear Solver (if you don't run expert, then you don't need this)
# ##########################################
# # RUN apt-get install libblas3gf libblas-dev liblapack3gf liblapack-dev gfortran
# # 1. Download the coinhsl package from http://www.hsl.rl.ac.uk/ipopt/
# # 2. Place it 
# # 2. This code will copy it into Docker and compile

# WORKDIR /home/kota/installations
# ADD coinhsl-archive-2021.05.05 /home/kota/installations/coinhsl-archive-2021.05.05
# WORKDIR /home/kota/installations/coinhsl-archive-2021.05.05
# #sudo make uninstall && sudo make clean #Only needed if you have installed it before
# RUN ./configure LIBS="-llapack" --with-blas="-L/usr/lib -lblas" CXXFLAGS="-g -O3 -fopenmp" FCFLAGS="-g -O3 -fopenmp" CFLAGS="-g -O3 -fopenmp" #the output should say `checking for metis to compile... yes`
# RUN make install #(the files will go to /usr/local/lib)
# WORKDIR /usr/local/lib
# RUN ln -s libcoinhsl.so libhsl.so #(This creates a symbolic link `libhsl.so` pointing to `libcoinhsl.so`). See https://github.com/casadi/casadi/issues/1437
# RUN printf '\nexport LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"' >> ~/.bashrc

# RUN pip3 install numpy==1.21.1
# RUN pip3 install python-dateutil==2.8.2
# RUN pip3 install imitation
# RUN pip3 install matplotlib
# RUN pip3 install numpy==1.25.0

# Useful things in ~/.bashrc
##########################################
RUN printf '\nsource /opt/ros/noetic/setup.bash' >> ~/.bashrc && \
      printf '\nalias cb="catkin build"' >> ~/.bashrc && \
      printf '\nalias gs="git status"' >> ~/.bashrc && \
      printf '\nalias gp="git push"' >> ~/.bashrc && \
      printf '\nalias kr="killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f panther & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill panther_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f panther_commands & pkill -f dynamic_corridor & tmux kill-server & pkill -f perfect_controller & pkill -f publish_in_gazebo"' >> ~/.bashrc && \
      printf '\nalias py="python3"' >> ~/.bashrc && \
      printf '\nalias tks="tmux kill-server"' >> ~/.bashrc

WORKDIR /home/kota/ws/src/deep_panther
