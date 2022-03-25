%This file plots the cost as a function of yaw for a given fixed spatial
%trajectory. It then uses the closed-form solution of this problem and
%plots it.

close all;  clear all; clc;

set(0,'defaulttextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');

import casadi.*
addpath(genpath('./../../submodules/minvo/src/utils'));
addpath(genpath('./../../submodules/minvo/src/solutions'));
addpath(genpath('./more_utils'));
addpath(genpath('./hopf_visualization'));

set(0,'DefaultFigureWindowStyle','docked') %'normal' 'docked'
set(0,'defaultfigurecolor',[1 1 1])
opti = casadi.Opti();


deg_pos=3;
deg_yaw=2;
num_seg =4; %number of segments
num_max_of_obst=10; %This is the maximum num of the obstacles 

basis="MINVO"; %MINVO OR B_SPLINE or BEZIER. This is the basis used for collision checking (in position, velocity, accel and jerk space), both in Matlab and in C++

y0=opti.parameter(1,1);
ydot0=opti.parameter(1,1); 
ydotf=opti.parameter(1,1);
alpha=opti.parameter(1,1); 

ydot0_n=ydot0*alpha;
ydotf_n=ydotf*alpha;

t0_n=0; 
tf_n=1.0;

dim_pos=3;
dim_yaw=1;

sp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.
% sy=MyClampedUniformSpline(t0_n,tf_n,deg_yaw, dim_yaw, num_seg, opti); %spline yaw.

n_samples=10;

all_t_n=linspace(t0_n,tf_n,n_samples);


all_w_fevar=MX.sym('all_w_fevar',3,n_samples);

all_yaw=[];
dataPoints={};
i=1;
for t=all_t_n
    
    pos=sp.getPosT(t);
%     a=sp.getAccelT(t);
    a_n = sp.getAccelT(t);
    a = a_n/(alpha^2);

    ee=all_w_fevar(:,i)-pos;
    xi=a+[0 0 9.81]';

    r0_star=(ee*norm(xi)^2 - (ee'*xi)*xi);
    r0_star=r0_star/norm(r0_star);
    
    tmp=pos+r0_star;

    qabc=qabcFromAccel(a, 9.81);
    Rabc=toRotMat(qabc);
    w_Tabc_b0y=[Rabc pos; 0 0 0 1]; %The frame "b0y" stands for "body zero yaw", and has yaw=0
    
    b0y_r0star=invPose(w_Tabc_b0y)*[tmp;1]; %express the vector in the frame "b0y" 
    b0y_r0star=b0y_r0star(1:3); 
%     assert(abs(b0y_r0star(3))<0.0001)
    
    dataPoints{end+1}=b0y_r0star(1:2)';
    all_yaw=[all_yaw atan2(b0y_r0star(2),b0y_r0star(1))]; %compute the yaw angle 
    
    i=i+1;
end


all_yaw_corrected=shiftToEnsureNoMoreThan2Pi(all_yaw);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%    FUNCTION TO FIT A SPLINE TO YAW SAMPLES     %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sy=MyClampedUniformSpline(t0_n,tf_n,deg_yaw, dim_yaw, num_seg, opti); 

cost_function=0;
for i=1:numel(all_t_n)
    cost_function = cost_function + (sy.getPosT(all_t_n(i))-all_yaw_corrected(i))^2; 
end

lambda1=MX.sym('lambda1',1,1);
lambda2=MX.sym('lambda2',1,1);
lambda3=MX.sym('lambda3',1,1);

%Note that y0 \equiv all_yaw(1)
c1= sy.getPosT(t0_n) - y0; %==0
c2= sy.getVelT(t0_n) - ydot0_n; %==0
c3= sy.getVelT(tf_n) - ydotf_n; %==0

lagrangian = cost_function  +  lambda1*c1 + lambda2*c2 + lambda3*c3;

variables=[sy.getCPsAsMatrix() lambda1 lambda2  lambda3];

kkt_eqs=jacobian(lagrangian, variables)'; %I want kkt=[0 0 ... 0]'

%Obtain A and b
b=-casadi.substitute(kkt_eqs, variables, zeros(size(variables))); %Note the - sign
A=jacobian(kkt_eqs, variables);

%solution=A\b;  %Option 1
solution=solve(A,b, 'symbolicqr'); %Option 2, allows for expand(), see https://groups.google.com/g/casadi-users/c/kOYv6lvNgEI/m/UiAYsdJEBAAJ

pCPs=sp.getCPsAsMatrix();
% yCPs=sy.getCPsAsMatrix();


f= Function('f', {pCPs, all_w_fevar, alpha, y0,ydot0, ydotf }, {solution(1:end-3),all_yaw_corrected}, ...
                 {'pCPs', 'all_w_fevar', 'alpha', 'y0','ydot0', 'ydotf'}, {'solution','all_yaw_corrected'} );

f=f.expand(); %

f.save('./casadi_generated_files/get_optimal_yaw_for_fixed_pos.casadi') 

pCPs_value=[   -4.0000   -4.0000   8    5    3   2     -4;
         -2         0         0   -5   -7   -8   -9;
         0         2         -2    7    0.0052    3    0.0052];

alpha_value=8.5;
y0_value=1.5;
ydot0_value=-2;
ydotf_value=-1;
all_w_fevar_value=([1 -2 3]').*ones(3,n_samples);

disp("Calling function")
tic
result=f('pCPs', pCPs_value, 'all_w_fevar', all_w_fevar_value, 'alpha', alpha_value, ...
        'y0', y0_value,'ydot0', ydot0_value, 'ydotf', ydotf_value);
toc
disp("Function called")

sy.updateCPsWithSolution(full(result.solution)')
sy.plotPos();
subplot(1,1,1); hold on;
% plot(all_t_n,full(result.all_yaw),'o')
plot(all_t_n,full(result.all_yaw_corrected),'o')

