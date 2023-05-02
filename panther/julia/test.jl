##
## Call Python script from Julia using MPI (~/.julia/bin/mpiexecjl -np 2 julia main.jl )
##

using MPI; MPI.install_mpiexecjl(force=true)
using Distributed
using PyCall

# parameters
ACT_SIZE = 22
OBST_SIZE = 43

# Initialize MPI
MPI.Init()

# Get the number of processes and the rank of this process
comm = MPI.COMM_WORLD
size = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# Define the Python script to run
# script = "/home/kota/Research/deep-panther_ws/src/deep_panther/panther/julia/call_expert.py"
# @pyinclude(script)
# obs, act = py"get_expert_action"()



# Gather the output from each process
act_buff = MPI.Gather(act,  comm; root=0)
obs_buff = MPI.Gather(obs,  comm; root=0)

# Finalize MPI
MPI.Finalize()

println("Output from process $rank ", act_buff)
println("Output from process $rank ", obs_buff)

# put the output into a file