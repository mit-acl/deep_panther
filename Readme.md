# Deep-PANTHER: Learning-Based Perception-Aware Trajectory Planner in Dynamic Environments #


[![Deep-PANTHER: Learning-Based Perception-Aware Trajectory Planner in Dynamic Environments](./panther/imgs/deep_panther.gif)](https://www.youtube.com/watch?v=53GBjP1jFW8 "Deep-PANTHER: Learning-Based Perception-Aware Trajectory Planner in Dynamic Environments")  

Deep-PANTHER deployed on different environments. The policy in all the videos above is the same one, and was trained using an obstacle that followed a trefoil-knot trajectory. The green pyramid represents the field of view of the camera. 

## Citation

When using Deep-PANTHER, please cite [Deep-PANTHER: Learning-Based Perception-Aware Trajectory Planner in Dynamic Environments](https://arxiv.org/abs/2209.01268) ([pdf](https://arxiv.org/pdf/2209.01268.pdf) and [video](https://www.youtube.com/watch?v=53GBjP1jFW8)):

```bibtex
@article{tordesillas2023deep,
  title={{Deep-PANTHER}: Learning-based perception-aware trajectory planner in dynamic environments},
  author={Tordesillas, Jesus and How, Jonathan P},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```

## General Setup

Deep-PANTHER has been tested with Ubuntu 20.04/ROS Noetic. Other Ubuntu/ROS version may need some minor modifications, feel free to [create an issue](https://github.com/mit-acl/panther/issues) if you have any problems.

The instructions below assume that you have ROS Noetic installed on your Linux machine.

### <ins>Dependencies<ins>

> Note: the instructions below are partly taken from [here](https://github.com/casadi/casadi/wiki/InstallationLinux#installation-on-linux)

#### IPOPT
```bash
sudo apt-get install gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends
sudo apt-get install coinor-libipopt1v5 coinor-libipopt-dev
```

#### CasADi
```bash
sudo apt-get remove swig swig3.0 swig4.0 #If you don't do this, the compilation of casadi may fail with the error "swig error : Unrecognized option -matlab"
mkdir ~/installations && cd ~/installations
git clone https://github.com/jaeandersson/swig
cd swig
git checkout -b matlab-customdoc origin/matlab-customdoc        
sh autogen.sh
sudo apt-get install gcc-7 g++-7 bison byacc
sudo apt-get install libpcre3 libpcre3-dev
./configure CXX=g++-7 CC=gcc-7            
make
sudo make install


cd ~/installations && mkdir casadi && cd casadi
git clone https://github.com/casadi/casadi
cd casadi 
#cd build && make clean && cd .. && rm -rf build #Only if you want to clean any previous installation/compilation 
mkdir build && cd build
cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON ..
#You may need to run the command above twice until the output says that `Ipopt` has been detected (although `IPOPT` is also being detected when you run it for the first time)
make -j20
sudo make install
```
#### Virtual Python environment
```bash
sudo apt-get install python3-venv
cd ~/installations && mkdir venvs_python && cd venvs_python 
python3 -m venv ./my_venv
printf '\nalias activate_my_venv="source ~/installations/venvs_python/my_venv/bin/activate"' >> ~/.bashrc
source ~/.bashrc
activate_my_venv
```

### <ins>Compilation<ins>
And finally download the repo and compile it:

```bash
sudo apt-get install git-lfs ccache 
cd ~/Desktop/
mkdir ws && cd ws && mkdir src && cd src
git clone https://github.com/mit-acl/deep_panther
cd deep_panther
git lfs install
git submodule init && git submodule update
cd panther_compression/imitation
pip install numpy Cython wheel seals rospkg defusedxml empy pyquaternion pytest
pip install -e .
sudo apt-get install python3-catkin-tools #To use catkin build
sudo apt-get install ros-"${ROS_DISTRO}"-rviz-visual-tools ros-"${ROS_DISTRO}"-pybind11-catkin ros-"${ROS_DISTRO}"-tf2-sensor-msgs ros-"${ROS_DISTRO}"-jsk-rviz-plugins
cd ~/Desktop/ws/
catkin build
printf '\nsource PATH_TO_YOUR_WS/devel/setup.bash' >> ~/.bashrc #Remember to change PATH_TO_YOUR_WS
printf '\nexport PYTHONPATH="${PYTHONPATH}:$(rospack find panther)/../panther_compression"' >> ~/.bashrc 
source ~/.bashrc
```

## Usage

Simply use:
```bash
roslaunch panther simulation.launch
```

Wait until the terminal says `Planner initialized`. Then, you can press G (or click the option 2D Nav Goal on the top bar of RVIZ) and click any goal for the drone. By default, `simulation.launch` will use the policy Hung_dynamic_obstacles.pt (which was trained with trefoil-knot trajectories). You can change the trajectory followed by the obstacle during testing using the `type_of_obst_traj` field of the launch file.

You can also use policies trained using a static obstacle. Simply change the field `student_policy_path` of `simulation.launch`. The available policies have the format `A_epsilon_B.pt`, where `A` is the algorithm used: Hungarian (i.e., LSA), RWTAc, or RWTAr. `B` is the epsilon used. Note that this epsilon is irrelevant for the LSA algorithm. Check the paper for further details. 


If you want to...

* **Use the expert:** You first need to install a linear solver (see instructions below). Then, you can use the expert by simply setting `use_expert: true`, `use_student: false` , and `pause_time_when_replanning:true` in `panther.yaml` and running `roslaunch panther simulation.launch`. 

* **Modify the optimization problem:**, You will need to have MATLAB installed (especifically, you will need the `Symbolic Math Toolbox` and the `Phased Array System Toolbox` installed), and follow the steps detailed in the MATLAB section below. You can then make any modification in the optimization problem by modifying the file `main.m`, and then running it. This will generate all the necessary `.casadi` files in the `casadi_generated_files` folder, which will be read by the C++ code.

* **Train the policy:** You first need to install a linear solver (see instructions below). Then, you can train a new policy but simply running `python3 policy_compression_train.py` inside the `panther_compression` folder. 


<details>
  <summary> <b>MATLAB (optional dependency)</b></summary>

First, when installing CasADi following the instructions above, you need to use `-DWITH_MATLAB=ON` instead of `-DWITH_MATLAB=OFF`. Then do the following:

```bash
#Open MATLAB, and type this:
edit(fullfile(userpath,'startup.m'))
#And in that file, add this line line 
addpath(genpath('/usr/local/matlab/'))
```

Now, you can restart Matlab (or run the file `startup.m`), and make sure this works:

```bash
import casadi.*
x = MX.sym('x')
disp(jacobian(sin(x),x))
```

</details>

<details>
  <summary> <b>Linear Solver (optional dependency)</b></summary>

Go to [http://www.hsl.rl.ac.uk/ipopt/](http://www.hsl.rl.ac.uk/ipopt/), click on `Personal Licence, Source` to install the solver `MA27` (free for everyone), and fill and submit the form. Once you receive the corresponding email, download the compressed file, uncompress it, and place it in the folder `~/installations` (for example). Then execute the following commands:

> Note: the instructions below follow [this](https://github.com/casadi/casadi/wiki/Obtaining-HSL) closely

```bash
cd ~/installations/coinhsl-2015.06.23
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz #This is the metis version used in the configure file of coinhsl
tar xvzf metis-4.0.3.tar.gz
#sudo make uninstall && sudo make clean #Only needed if you have installed it before
./configure LIBS="-llapack" --with-blas="-L/usr/lib -lblas" CXXFLAGS="-g -O3 -fopenmp" FCFLAGS="-g -O3 -fopenmp" CFLAGS="-g -O3 -fopenmp" #the output should say `checking for metis to compile... yes`
sudo make install #(the files will go to /usr/local/lib)
cd /usr/local/lib
sudo ln -s libcoinhsl.so libhsl.so #(This creates a symbolic link `libhsl.so` pointing to `libcoinhsl.so`). See https://github.com/casadi/casadi/issues/1437
echo "export LD_LIBRARY_PATH='\${LD_LIBRARY_PATH}:/usr/local/lib'" >> ~/.bashrc
```

<details>
  <summary> <b>Note</b></summary>

We recommend to use `MA27`. Alternatively, you can install both `MA27` and `MA57` by clicking on `Coin-HSL Full (Stable) Source` (free for academia) in [http://www.hsl.rl.ac.uk/ipopt/](http://www.hsl.rl.ac.uk/ipopt/) and then following the instructions above. Other alternative is to use the default `mumps` solver (no additional installation required), but its much slower than `MA27` or `MA57` You can change the linear solver used by changing the name of `linear_solver_name` in the file `main.m` and run that file.

Moreover, when using a linear solver different from `mumps`, you may need to start Matlab from the terminal (typing `matlab`). More info [in this issue](https://github.com/casadi/casadi/issues/2032). 

</details>

</details>

<details>
  <summary><b>Known Issues</b></summary>

  <h3>Missing PCL visualizations library</h3>

  When launching the simulation using `roslaunch panther simulation.launch` the following error might occur
  
  ```
  PluginlibFactory: The plugin for class 'jsk_rviz_plugin/TFTrajectory' failed to load.  Error: Failed to load library /opt/ros/noetic/lib//libjsk_rviz_plugins.so. Make sure that you are calling the PLUGINLIB_EXPORT_CLASS macro in the library code, and that names are consistent between this macro and your XML. Error string: Could not load library (Poco exception = libpcl_visualization.so.1.10: cannot open shared object file: No such file or directory)
  ```

  This is a version problem with `libpcl_visualization.so`. The correct version is `1.10.0+dfsg-5ubuntu1`.

  ```bash
  sudo apt-get update
  sudo apt-get install libvtk7.1p

  sudo apt install libpcl-apps1.10=1.10.0+dfsg-5ubuntu1 libpcl-common1.10=1.10.0+dfsg-5ubuntu1 libpcl-features1.10=1.10.0+dfsg-5ubuntu1 libpcl-filters1.10=1.10.0+dfsg-5ubuntu1 libpcl-io1.10=1.10.0+dfsg-5ubuntu1 libpcl-kdtree1.10=1.10.0+dfsg-5ubuntu1 libpcl-keypoints1.10=1.10.0+dfsg-5ubuntu1 libpcl-ml1.10=1.10.0+dfsg-5ubuntu1 libpcl-octree1.10=1.10.0+dfsg-5ubuntu1 libpcl-outofcore1.10=1.10.0+dfsg-5ubuntu1 libpcl-people1.10=1.10.0+dfsg-5ubuntu1 libpcl-recognition1.10=1.10.0+dfsg-5ubuntu1 libpcl-registration1.10=1.10.0+dfsg-5ubuntu1 libpcl-sample-consensus1.10=1.10.0+dfsg-5ubuntu1 libpcl-search1.10=1.10.0+dfsg-5ubuntu1 libpcl-segmentation1.10=1.10.0+dfsg-5ubuntu1 libpcl-stereo1.10=1.10.0+dfsg-5ubuntu1 libpcl-surface1.10=1.10.0+dfsg-5ubuntu1 libpcl-tracking1.10=1.10.0+dfsg-5ubuntu1 libpcl-visualization1.10=1.10.0+dfsg-5ubuntu1 libpcl-dev=1.10.0+dfsg-5ubuntu1

  sudo apt install ros-noetic-pcl-ros
  ```

  Note, that `sudo apt-get upgrade` will upgrade `libpcl` again and the error with `TFTrajectory` returns. To avoid this, set the package on hold

  ```bash
  sudo apt-mark hold libpcl-apps1.10=1.10.0+dfsg-5ubuntu1 libpcl-common1.10=1.10.0+dfsg-5ubuntu1 libpcl-features1.10=1.10.0+dfsg-5ubuntu1 libpcl-filters1.10=1.10.0+dfsg-5ubuntu1 libpcl-io1.10=1.10.0+dfsg-5ubuntu1 libpcl-kdtree1.10=1.10.0+dfsg-5ubuntu1 libpcl-keypoints1.10=1.10.0+dfsg-5ubuntu1 libpcl-ml1.10=1.10.0+dfsg-5ubuntu1 libpcl-octree1.10=1.10.0+dfsg-5ubuntu1 libpcl-outofcore1.10=1.10.0+dfsg-5ubuntu1 libpcl-people1.10=1.10.0+dfsg-5ubuntu1 libpcl-recognition1.10=1.10.0+dfsg-5ubuntu1 libpcl-registration1.10=1.10.0+dfsg-5ubuntu1 libpcl-sample-consensus1.10=1.10.0+dfsg-5ubuntu1 libpcl-search1.10=1.10.0+dfsg-5ubuntu1 libpcl-segmentation1.10=1.10.0+dfsg-5ubuntu1 libpcl-stereo1.10=1.10.0+dfsg-5ubuntu1 libpcl-surface1.10=1.10.0+dfsg-5ubuntu1 libpcl-tracking1.10=1.10.0+dfsg-5ubuntu1 libpcl-visualization1.10=1.10.0+dfsg-5ubuntu1 libpcl-dev=1.10.0+dfsg-5ubuntu1
  ```

  <h3>MA27 Installation</h3>

  Note, that on the HSL website you can download either the source code or the compiled libraries. Make sure to download the **source code**.

  If the steps to install the MA27 above do not work and you still get an error that MA27 cannot be found you might have to do the following:

  ```bash
  cd ~/installations
  git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
  cd ThirdParty-HSL
  ```

  Now move the Coin-HSL source code to `~/installations/ThirdParty-HSL/coinhsl`. Then

  ```bash
  cd ~/installations/ThirdParty-HSL
  ./configure
  make
  sudo make install
  ```

  These instructions are taken from [here](https://coin-or.github.io/Ipopt/INSTALL.html).
</details>
