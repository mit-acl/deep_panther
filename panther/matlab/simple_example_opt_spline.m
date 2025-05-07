close all; clc;clear;
doSetup();
import casadi.*

opti = casadi.Opti();

t0_n=0.0; 
tf_n=1.0;
total_time=2;
total_time_n=(tf_n-t0_n);
alpha=total_time/total_time_n;  %Please read explanation_normalization.svg

dim_pos=3;
deg_pos=3;
num_seg=5;
basis="MINVO";

sp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.

constraints=[];

%Initial conditions
constraints{end+1}= sp.getPosT(t0_n)== [0 1 1]' ;
constraints{end+1}= sp.getVelT(t0_n)== [-3 0 -0.4]' ;
constraints{end+1}= sp.getAccelT(t0_n)== [0 0 0]' ;

%Final conditions
constraints{end+1}= sp.getPosT(tf_n)== [1 2 3]' ;
constraints{end+1}= sp.getVelT(tf_n)== [0 0 0]' ;
constraints{end+1}= sp.getAccelT(tf_n)== [0 0 0]' ;

%Dynamic limits
v_max=2*ones(1,3);
a_max=7*ones(1,3);
j_max=50*ones(1,3);

%Dynamic limits normalized
v_max_n=v_max*alpha;
a_max_n=a_max*(alpha^2);
j_max_n=j_max*(alpha^3);

constraints=[constraints sp.getMaxVelConstraints(basis, v_max_n)];      %Max vel constraints (position)
constraints=[constraints sp.getMaxAccelConstraints(basis, a_max_n)];    %Max accel constraints (position)
constraints=[constraints sp.getMaxJerkConstraints(basis, j_max_n)];     %Max jerk constraints (position)

opts = struct;
opts.expand=true; %When this option is true, it goes WAY faster!
opts.print_time=0;
opts.ipopt.print_level=3; 
opts.ipopt.max_iter=500;
opti.solver('ipopt',opts); %{"ipopt.hessian_approximation":"limited-memory"} 

opti.subject_to(constraints)
opti.minimize(sp.getControlCost())

sol = opti.solve();

sp.updateCPsWithSolution(sol.value(sp.getCPsAsMatrix()))

sp.plotPosVelAccelJerk(v_max_n, a_max_n, j_max_n)
sp.plotPos3D()
