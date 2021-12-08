close all; clc;clear;
doSetup();
import casadi.*

opti = casadi.Opti();

t0_n=0.0; 
tf_n=1.0;

dim_pos=3;
deg_pos=3;
num_seg=6;
basis="MINVO"

sp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.

CPs=[   -4.0000   -4.0000   -4.0000   -1.7605   -0.1210    1.5184    3.7580    3.7580    3.7580;
         0         0         0   -2.1199   -1.8860   -2.2088   -0.0000   -0.0000   -0.0000;
         0         0         0    0.3969   -0.2470    0.4376    0.0000    0.0000    0.0000];

sp.updateCPsWithSolution(CPs)

sp.plotVel3D()

MV_cpoints=sp.getCPs_XX_Vel_ofInterval(basis, 2);

radius=0.4;
plotSphere(MV_cpoints{1},radius, 'r');
plotSphere(MV_cpoints{2},radius, 'g');
plotSphere(MV_cpoints{3},radius, 'b');