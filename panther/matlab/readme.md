Instructions to install Casadi from source, with Gurobi and from Matlab [April 2021], funciona:

NOTA: These instructions will make use of IPOPT 3.11.9 (the one installed by defaut when using `sudo apt-get install coinor-libipopt-dev` in Ubuntu 18.04). However, the binaries that Casadi provides have Ipopt 3.12.3 (see [this](https://github.com/casadi/casadi/releases#:~:text=3.12.3)). For a specific optimization problem I had, it worked perfectly when using Casadi from binaries (which uses IPOPT 3.12.3), but it reported `Restoration phase is called at point that is almost feasible` or `Restoration failed` when using Casadi from source (which uses IPOPT 3.11.9)


## Preliminaries
Follow the first section [here](https://github.com/casadi/casadi/wiki/InstallationLinux#installation-on-linux)

## Install IPOPT

### From source. 
Follow [this](https://github.com/casadi/casadi/wiki/InstallationLinux#option-2-compiling-ipopt-from-sources). But, if you wanna use MLK, use this configure command instead:
```
../configure --prefix=/usr/local --with-blas="-L$MKLROOT/lib/intel64 -lmkl_rt -liomp5 -lpthread -lm -ldl" ADD_FFLAGS=-fPIC ADD_CFLAGS=-fPIC ADD_CXXFLAGS=-fPIC
```

### From binaries.
`sudo apt-get install coinor-libipopt1v5`

You can check if this package is installed (using  `apt-get install`) by running `dpkg -l | grep ipopt`

## Install Casadi
These instructions partly follow [this](https://github.com/casadi/casadi/wiki/matlab):
```
sudo apt-get remove swig
sudo apt-get remove swig3.0  #If you don't do this, the compilation of casadi may fail with the error "swig error : Unrecognized option -matlab"
cd ~/installations
git clone https://github.com/jaeandersson/swig
cd swig
git checkout -b matlab-customdoc origin/matlab-customdoc        
sh autogen.sh
./configure CXX=g++-7 CC=gcc-7            
sudo apt-get install bison -y
sudo apt-get install byacc -y
make
sudo make install

#Clone the casadi github repo into ~/installations/casadi/ 
cd ~/installations/casadi/ 
Add gurobi91 to the file cmake/FindGUROBI.cmake   #Or in general to the gurobi version you have (type gurobi.sh to find this)
mkdir build && cd build
make clean 
cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=ON -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON -DWITH_GUROBI=ON ..
#For some reason, I needed to run the command above twice until `Ipopt` was detected (although `IPOPT` was being detected already)
make
sudo make install
```

See also https://github.com/jtorde/casadi  for some modifications I did!!

Then, type matlab in the terminal, run `edit(fullfile(userpath,'startup.m'))`, and add this line to that file:

addpath(genpath('/usr/local/matlab/')) 


Everything should work now:
```
import casadi.*
x = MX.sym('x')
disp(jacobian(sin(x),x))
```


===

I've modified the casadi source files to be able to do directly opti.change_problem_type('nlp') (or 'gurobi'), see also https://github.com/casadi/casadi/issues/2775

=======================================
