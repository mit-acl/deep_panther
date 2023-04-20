##
## you can test this script by running the following command
## ~/.julia/bin/mpiexecjl -np 2 julia test.jl
## -np 2 means that you are using 2 processes
##

using MPI; MPI.install_mpiexecjl(force=true)
using Distributed
using PyCall

# Initialize MPI
MPI.Init()

# Get the number of processes and the rank of this process
comm = MPI.COMM_WORLD
size = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# Define the Python script to run
script = "/home/kota/Research/deep-panther_ws/src/deep_panther/panther/julia/test.py"
@pyinclude(script)
tmp = py"main"(rank)

# Gather the output from each process
recv_buff = MPI.Gather(tmp,  comm; root=0)
if rank == 0
    println("Output from process $rank ", recv_buff)
end

# Finalize MPI
MPI.Finalize()