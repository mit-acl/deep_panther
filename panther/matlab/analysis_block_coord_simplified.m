% This files implements block-coordinate descent on yaw and pos*/

close all; clc;clear;



doSetup();

import casadi.*
% 
% all_comp_times_t_wall=[];
% all_comp_times_t_proc=[];
% 
% %%
% close all;
% subplot(2,1,1);
% histogram(1000*all_comp_times_t_proc,numel(all_comp_times_t_proc)); title ('Proc time'); xlabel('time(ms)')
% subplot(2,1,2);
% histogram(1000*all_comp_times_t_wall,numel(all_comp_times_t_wall)); title ('Wall time'); xlabel('time(ms)')
% 
% fprintf('Mean (ms), t_proc=%.2f ms\n', mean(1000*all_comp_times_t_proc))
% fprintf('Mean (ms), t_wall=%.2f ms\n', mean(1000*all_comp_times_t_wall))
% for novale=1:20

opti = casadi.Opti();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% CONSTANTS! %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

use_yaw_closed_form=true; %If false, yaw will be optimized.

deg_pos=3;
deg_yaw=2;
num_seg =6; %number of segments
num_max_of_obst=1; %This is the maximum num of the obstacles 
num_samples_simpson=7;  %This will also be the num_of_layers in the graph yaw search of C++
num_of_yaw_per_layer=40; %This will be used in the graph yaw search of C++
                         %Note that the initial layer will have only one yaw (which is given) 
basis="MINVO"; %MINVO OR B_SPLINE or BEZIER. This is the basis used for collision checking (in position, velocity, accel and jerk space), both in Matlab and in C++
linear_solver_name='ma27'; %mumps [default, comes when installing casadi], ma27, ma57, ma77, ma86, ma97 
print_level=5; %From 0 (no verbose) to 12 (very verbose), default is 5
delta_t=1.0; %Normalized duration of each segment
t0_n=0;   tf_n=delta_t*num_seg; %t0 and tf normalized

dim_pos=3;  dim_yaw=1;

offset_vel=0.1;

assert(tf_n>t0_n);

const_p={}; const_y={};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% PARAMETERS! %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% DEFINITION
%%%%% factors for the cost
c_pos_smooth=            opti.parameter(1,1);
c_yaw_smooth=             opti.parameter(1,1);
c_fov=  opti.parameter(1,1);
c_final_pos = opti.parameter(1,1);
c_final_yaw = opti.parameter(1,1);
c_total_time = opti.parameter(1,1);
% c_costs.dist_im_cost=         opti.parameter(1,1);

Ra=opti.parameter(1,1);

thetax_FOV_deg=opti.parameter(1,1);    %total angle of the FOV in the x direction
thetay_FOV_deg=opti.parameter(1,1);    %total angle of the FOV in the y direction

thetax_half_FOV_deg=thetax_FOV_deg/2.0; %half of the angle of the cone
thetax_half_FOV_rad=thetax_half_FOV_deg*pi/180.0;

thetay_half_FOV_deg=thetay_FOV_deg/2.0; %half of the angle of the cone
thetay_half_FOV_rad=thetay_half_FOV_deg*pi/180.0;

%total_time=opti.parameter(1,1); %This allows a different t0 and tf than the one above

alpha=opti.variable(1,1); %Total time is (tf_n-t0_n)*alpha. 
total_time=alpha*(tf_n-t0_n);

% scaling=(tf_n-t0_n)/total_time;

%%%%% Initial and final conditions, and max values
%FOR POSITION
p0=opti.parameter(3,1); v0=opti.parameter(3,1); a0=opti.parameter(3,1);
pf=opti.parameter(3,1); vf=opti.parameter(3,1); af=opti.parameter(3,1);

v_max=opti.parameter(3,1);
a_max=opti.parameter(3,1);
j_max=opti.parameter(3,1);

%Normalized v0, a0, v_max,...
v0_n=v0*alpha;
a0_n=a0*(alpha^2);
vf_n=vf*alpha;
af_n=af*(alpha^2);
v_max_n=v_max*alpha;
a_max_n=a_max*(alpha^2); 
j_max_n=j_max*(alpha^3);

%FOR YAW
if(use_yaw_closed_form==false)
    y0=opti.parameter(1,1); ydot0=opti.parameter(1,1); 
    yf=opti.parameter(1,1); ydotf=opti.parameter(1,1);
    ydot_max=opti.parameter(1,1);
    
    ydot0_n=ydot0*alpha;
    ydotf_n=ydotf*alpha;
    ydot_max_n=ydot_max*alpha; %v_max for yaw
    
end

%%%%% Planes
n={}; d={};
for i=1:(num_max_of_obst*num_seg)
    n{i}=opti.parameter(3,1); 
    d{i}=opti.parameter(1,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% CREATION OF THE SPLINES! %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.

if(use_yaw_closed_form==false)
    sy=MyClampedUniformSpline(t0_n,tf_n,deg_yaw, dim_yaw, num_seg, opti); %spline yaw.
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% CONSTRAINTS! %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%Initial and final conditions for POSITION
const_p{end+1}=sp.getPosT(t0_n)== p0;
const_p{end+1}=sp.getVelT(t0_n)== v0_n;
const_p{end+1}=sp.getAccelT(t0_n)== a0_n;
% opti.subject_to( sp.getPosT(tf)== pf );
const_p{end+1}=sp.getVelT(tf_n)== vf_n;
const_p{end+1}=sp.getAccelT(tf_n)== af_n;

%Dynamic limits for POSITION
const_p=[const_p sp.getMaxVelConstraints(basis, v_max_n)];      %Max vel constraints (position)
const_p=[const_p sp.getMaxAccelConstraints(basis, a_max_n)];    %Max accel constraints (position)
const_p=[const_p sp.getMaxJerkConstraints(basis, j_max_n)];     %Max jerk constraints (position)

if(use_yaw_closed_form==false)
    %%Initial and final conditions for POSITION
    const_y{end+1}=sy.getPosT(t0_n)== y0;
    const_y{end+1}=sy.getVelT(t0_n)== ydot0_n ;
    const_y{end+1}=sy.getVelT(tf_n)==ydotf_n; % Needed: if not (and if you are minimizing ddyaw), dyaw=cte --> yaw will explode
    
    %Dynamic limits for YAW
    const_y=[const_y sy.getMaxVelConstraints(basis, ydot_max_n)];   %Max vel constraints (yaw)
end


%Obstacle constraints;
for j=1:(sp.num_seg)

    %Get the control points of the interval
    Q=sp.getCPs_XX_Pos_ofInterval(basis, j);

    %Plane constraints
    for obst_index=1:num_max_of_obst
      ip = (obst_index-1) * sp.num_seg + j;  % index plane
           
      %and the control points on the other side
      for kk=1:size(Q,2)
        const_p{end+1}= n{ip}'*Q{kk} + d{ip} <= 0;
      end
    end  
end



g=9.81;
%Compute perception cost
dist_im_cost=0;
vel_im_cost=0;
fov_cost=0;

clear i
t_simpson_n=linspace(t0_n,tf_n,num_samples_simpson);
h=alpha*(t_simpson_n(2)-t_simpson_n(1));


% u=MX.sym('u',1,1); %it must be defined outside the loop (so that then I can use substitute it regardless of the interval
w_fevar=MX.sym('w_fevar',3,1); %it must be defined outside the loop (so that then I can use substitute it regardless of the interval
w_velfewrtworldvar=MX.sym('w_velfewrtworld',3,1);
% yaw= MX.sym('yaw',1,1);  
simpson_index=1;
simpson_coeffs=[];
all_fov_costs=[];
all_simpson_constants=[];
all_target_isInFOV=[];

f=0.05;%focal length in meters


for t_n=t_simpson_n
    
    w_t_b = sp.getPosT(t_n);
    a=sp.getAccelT(t_n)/(alpha^(2));
    xi=a+[0;0;g];
    
    if(use_yaw_closed_form==false)
  
        xi1=xi(1); xi2=xi(2); xi3=xi(3);
        nxi2= xi1*xi1 + xi2*xi2 + xi3*xi3;
        nxi=sqrt(nxi2);    
        nxi2_plus_xi3nxi=nxi2 + xi(3)*nxi;
        
        yaw=sy.getPosT(t_n);
        b1= [ (1-(xi1^2)/nxi2_plus_xi3nxi)*cos(yaw) -    xi1*xi2*sin(yaw)/nxi2_plus_xi3nxi;
              -xi1*xi2*cos(yaw)/nxi2_plus_xi3nxi     +    (1 - (xi2^2)/nxi2_plus_xi3nxi)*sin(yaw);
               (xi1/nxi)*cos(yaw)                 +    (xi2/nxi)*sin(yaw)         ]; %note that, by construction, norm(b1)=1;
    
    else
    
          b1=optimalb1FromPosPosFeatureAndAccel(w_t_b, zeros(3,1), a);
    end
    
    w_e=-w_t_b; %Asumming feature is in [0 0 0]';
    
    f=cos(thetax_half_FOV_deg*pi/180.0) - b1'*w_e/norm(w_e); %Constraint is f<=0
    
    fov_cost_j=max(0,f)^3; %Penalty associated with the constraint
    
    simpson_constant=(h/3.0)*getSimpsonCoeff(simpson_index,num_samples_simpson);
    
    fov_cost=fov_cost + simpson_constant*fov_cost_j; %See https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule
    
    all_fov_costs=[all_fov_costs fov_cost_j];
    
    simpson_index=simpson_index+1;
    all_simpson_constants=[all_simpson_constants simpson_constant];
    
end

%At this point, fov_cost must be in [0, sum(all_simpson_constants)]
%Let's normalize it in [0,1]
fov_cost=fov_cost/sum(all_simpson_constants);  %This is to make sure that this term is in [0,1]

%Cost
pos_smooth_cost=sp.getControlCost()/(alpha^(sp.p-1)); %This is integral of jerk^2 over total_time=alpha*(tf_n-t0_n);
pos_smooth_cost=pos_smooth_cost/((j_max'*j_max)*total_time); %This is to make sure that this term is in [0,1]


final_pos_cost=(sp.getPosT(tf_n)- pf)'*(sp.getPosT(tf_n)- pf);
total_time_cost=alpha*(tf_n-t0_n);

total_cost=c_pos_smooth*pos_smooth_cost+...
           c_fov*fov_cost+...
           c_final_pos*final_pos_cost+...
           c_total_time*total_time_cost;

if(use_yaw_closed_form==false)
    yaw_smooth_cost=sy.getControlCost()/(alpha^(sy.p-1));
    final_yaw_cost=(sy.getPosT(tf_n)- yf)^2;
 
    total_cost=total_cost+ c_yaw_smooth*yaw_smooth_cost + c_final_yaw*final_yaw_cost;
    
end
    
opti.minimize(simplify(total_cost));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLVE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

all_nd=[];
for i=1:(num_max_of_obst*num_seg)
    all_nd=[all_nd [n{i};d{i}]];
end

pCPs=sp.getCPsAsMatrix();
%  pCPs_par=opti.parameter(3,size(pCPs,2));

v_max_value=3*ones(3,1);
a_max_value=4*ones(3,1);
j_max_value=10*ones(3,1);

alpha_value=1.0;
thetax_FOV_deg_value=30;
thetay_FOV_deg_value=30;
Ra_value=12.0;

if(use_yaw_closed_form==false)
    
    yCPs=sy.getCPsAsMatrix();
%     yCPs_par=opti.parameter(1,size(yCPs,2));
   
    
    y0_value=0.0;
    yf_value=0.0;
    ydot0_value=0.0;
    ydotf_value=0.0;
    ydot_max_value=1.0;
end

p0_value=[-4;0.0; 0.0];
v0_value=[0;0;0];
a0_value=[0;0;0];

pf_value=[4.0;0.0; 0.0];
vf_value=[0;0;0];
af_value=[0;0;0];

% all_nd_value= zeros(4,num_max_of_obst*num_seg);
all_nd_value=[];
for j=1:floor(num_seg/2)
    all_nd_value=[all_nd_value (1/sqrt(2))*[1; 1; 0; 2] ];
end

for j=(floor(num_seg/2)+1):num_seg
    all_nd_value=[all_nd_value (1/sqrt(2))*[-1; 1; 0; 2] ];
end

tmp1=[   -4.0000   -4.0000   -4.0000    0.7111 1.0 2.0   4         4          4;
         0         0         0           0.5     1.0    0.5  0           0         0;
         0         0         0             0    0   0    0           0         0];
     
tmp2=[   -0.0000   -0.0000    0.2754  1.0  1.5  2.1131    2.6791    2.6791];


% tmp1=rand(size(tmp1));
% tmp2=rand(size(tmp2));

% 
% tmp1=[   -4.0000   -4.0000   -4.0000  -23.7331  -12.3799   -9.7726    1.8451    1.8451    1.8451
%          0         0    0.0000  -13.0382  -33.4583  -31.3069  -21.0817  -21.0817  -21.0817
%   -10.0000  -10.0000  -10.0000   15.0714   27.1907   20.4458   -2.0376   -2.0376   -2.0376];
% 
% tmp2=[ 0 0 0 0 0 0 0 0];

all_params_and_init_guesses=[...
              {createStruct('thetax_FOV_deg', thetax_FOV_deg, thetax_FOV_deg_value)},...
              {createStruct('thetay_FOV_deg', thetay_FOV_deg, thetay_FOV_deg_value)},...
              {createStruct('Ra', Ra, Ra_value)},...
              {createStruct('p0', p0, p0_value)},...
              {createStruct('v0', v0, v0_value)},...
              {createStruct('a0', a0, a0_value)},...
              {createStruct('pf', pf, pf_value)},...
              {createStruct('vf', vf, vf_value)},...
              {createStruct('af', af, af_value)},...
              {createStruct('v_max', v_max, v_max_value)},...
              {createStruct('a_max', a_max, a_max_value)},...
              {createStruct('j_max', j_max, j_max_value)},...
              {createStruct('all_nd', all_nd, all_nd_value)},...
              {createStruct('c_pos_smooth', c_pos_smooth, 0.01)},...
              {createStruct('c_fov', c_fov, 1.0)},...
              {createStruct('c_final_pos', c_final_pos, 10.0)},... 
              {createStruct('c_total_time', c_total_time, 100.0)},...
              {createStruct('alpha', alpha, alpha_value)},...
              {createStruct('pCPs', pCPs, tmp1)}];
          
if(use_yaw_closed_form==false)
    
    all_params_and_init_guesses=[...
             {createStruct('y0', y0, y0_value)},...
             {createStruct('ydot0', ydot0, ydot0_value)},...
             {createStruct('yf', yf, yf_value)},...
             {createStruct('ydotf', ydotf, ydotf_value)},...
             {createStruct('ydot_max', ydot_max, ydot_max_value)},... 
             {createStruct('yCPs', yCPs, tmp2)},...
             {createStruct('c_final_yaw', c_final_yaw, 0.0)},...
             {createStruct('c_yaw_smooth', c_yaw_smooth, 0.0)},...
             all_params_and_init_guesses];
    
end

%{createStruct('yCPs_par', yCPs_par, tmp2)},...
%{createStruct('pCPs_par', pCPs_par, tmp1)}

vars=[];
names=[];
names_value={};
for i=1:numel(all_params_and_init_guesses)
    vars=[vars {all_params_and_init_guesses{i}.param}];
    names=[names {all_params_and_init_guesses{i}.name}];

    names_value{end+1}=all_params_and_init_guesses{i}.name;
    names_value{end+1}=double2DM(all_params_and_init_guesses{i}.value); 

end


opts = struct;
opts.expand=true; %When this option is true, it goes WAY faster!
opts.print_time=true;
opts.ipopt.print_level=print_level; 
%opts.ipopt.print_frequency_iter=1e10;%1e10 %Big if you don't want to print all the iteratons
opts.ipopt.linear_solver=linear_solver_name;
opts.jit=true;%If true, when I call solve(), Matlab will automatically generate a .c file, convert it to a .mex and then solve the problem using that compiled code
opts.compiler='shell';
opts.jit_options.flags='-Ofast';  %Takes ~15 seconds to generate if O0 (much more if O1,...,O3)
opts.jit_options.verbose=true;  %See example in shallow_water.cpp
% opts.ipopt.hessian_approximation='limited-memory';
% opts.ipopt.line_search_method='cg-penalty';
% opts.ipopt.accept_every_trial_step='yes';
% opts.ipopt.alpha_for_y='max';
opti.solver('ipopt',opts); %{"ipopt.hessian_approximation":"limited-memory"} 



opti.subject_to([const_p, const_y])

results_vars={pCPs, total_cost, pos_smooth_cost, alpha, fov_cost, final_pos_cost};
results_names={'pCPs','total_cost','pos_smooth_cost','alpha','fov_cost','final_pos_cost'};

if(use_yaw_closed_form==false)
    results_vars=[results_vars {yCPs,  yaw_smooth_cost, final_yaw_cost}];
    results_names=[results_names {'yCPs', 'yaw_smooth_cost', 'final_yaw_cost'}];
end

my_func = opti.to_function('my_func', vars, results_vars, names, results_names);
sol=my_func( names_value{:});
full(sol.pCPs)

all_var_solved=[{createStruct('pCPs', pCPs, full(sol.pCPs))},...
                {createStruct('alpha', alpha, full(sol.alpha))}];
            
if(use_yaw_closed_form==false)
    all_var_solved=[{createStruct('yCPs', yCPs, full(sol.yCPs))},...
                    all_var_solved];
end
            
if(use_yaw_closed_form==false)
  full(sol.yCPs)
end

cprintf('Green','Total time trajec=%.2f s (alpha=%.2f) \n', full(sol.alpha*(tf_n-t0_n)), full(sol.alpha)    )


[t_proc_total, t_wall_total]= timeInfo(my_func);


% %%
% sp_normalized=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.
% sp_normalized.updateCPsWithSolution(full(sol.pCPs))
% 
% sp_real=MyClampedUniformSpline(t0_n,full(sol.alpha)*tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.
% sp_real.updateCPsWithSolution(full(sol.pCPs))
% 
% real_cost=sp_real.getControlCost();
% 
% norm_cost=sp_normalized.getControlCost();
% 
% %norm_cost=real_cost*alpha^(sp.p-1)
% assert(abs(norm_cost-real_cost*full(sol.alpha)^(2*sp.p-1))<1e-7)
% 
% % sp_normalized.getControlCost()/sp_real.getControlCost()


sp_cpoints_var=sp.getCPsAsMatrix();



%Store solution
sp.updateCPsWithSolution(full(sol.pCPs))
if(use_yaw_closed_form==false)
    sy_cpoints_var=sy.getCPsAsMatrix();
    sy.updateCPsWithSolution(full(sol.yCPs))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOTTING! %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha_result=full(sol.alpha);
v_max_value_n=v_max_value*alpha_result;
a_max_value_n=a_max_value*alpha_result^2;
j_max_value_n=j_max_value*alpha_result^3;

sp.plotPosVelAccelJerk(v_max_value_n, a_max_value_n, j_max_value_n)

% % sp.plotPosVelAccelJerkFiniteDifferences();
% sy.plotPosVelAccelJerk(ydot_max_value)
% sy.plotPosVelAccelJerkFiniteDifferences();

sp.plotPos3D();
plotSphere( sp.getPosT(t0_n),0.2,'b'); plotSphere( sp.getPosT(tf_n),0.2,'r'); 

view([280,15]); axis equal
% 
disp("Plotting")
for t_i=t_simpson_n %t0:0.3:tf  
    
    w_t_b = sp.getPosT(t_i);
    accel = sp.getAccelT(t_i)/(alpha_result^2);
    
    %Obtain w_R_b from accel and psi
    if(use_yaw_closed_form==false)
        yaw = sy.getPosT(t_i);
        qabc=qabcFromAccel(accel, 9.81);
        qpsi=[cos(yaw/2), 0, 0, sin(yaw/2)]; %Note that qpsi has norm=1
        q=multquat(qabc,qpsi); %Note that q is guaranteed to have norm=1
        w_R_b=toRotMat(q);
    else
        xi=accel+[0;0;g];
        b3=xi/norm(xi);
        b1=optimalb1FromPosPosFeatureAndAccel(w_t_b, zeros(3,1), accel);
        b2=cross(b3, b1);
        w_R_b=[b1 b2 b3];
        assert(norm(w_R_b'*w_R_b-eye(3))<1e-6)
    end
    
    w_T_b=[w_R_b w_t_b; 0 0 0 1];
    plotAxesArrowsT(0.5,w_T_b)
    
    %Plot the FOV cone
    b_T_c_value= [roty(90)*rotz(-90) zeros(3,1); zeros(1,3) 1];
    w_T_c=w_T_b*b_T_c_value;
    position=w_T_c(1:3,4);
    direction=w_T_c(1:3,3);
    length=1;
    plotCone(position,direction,thetax_FOV_deg_value,length); 

end

plotSphere(zeros(3,1),0.2,'g');

% for i=1:num_samples_simpson
%     plotSphere(all_w_fe_value(:,i),0.2,'g');
% end

grid on; xlabel('x'); ylabel('y'); zlabel('z'); camlight; lightangle(gca,45,0)


syms x y z real
for i=1:size(all_nd_value,2)
   fimplicit3(all_nd_value(:,i)'*[x;y;z;1],[-4 4 -4 4 -2 2], 'MeshDensity',2, 'FaceAlpha',0.6) 
end

view(-91,90)



all_fov_costs_evaluated=substituteWithSolution(all_fov_costs, all_var_solved, all_params_and_init_guesses);

figure;
plot(all_fov_costs_evaluated,'-o'); title('Fov cost. $>$0 means not in FOV')

%%

% 
% all_comp_times_t_proc(end+1)=t_proc_total;
% all_comp_times_t_wall(end+1)=t_wall_total;


% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% BLOCK COORDINATE DESCENT [works, but select use_yaw_closed_form=true]
% 
% import casadi.*
% clc
% opti_p=opti.copy;
% 
% %CREATE SOLVER FOR POSITION, YAW IS FIXED 
% %See https://github.com/casadi/casadi/wiki/FAQ:-how-to-perform-jit-for-function-evaluations-of-my-optimization-problem%3F
% opts.jit=true;%If true, when I call solve(), Matlab will automatically generate a .c file, convert it to a .mex and then solve the problem using that compiled code
% opts.compiler='shell';
% opts.jit_options.flags='-O3';  %Takes ~15 seconds to generate if O0 (much more if O1,...,O3)
% opts.jit_options.verbose=true;  %See example in shallow_water.cpp
% opti.solver('ipopt',opts); 
% 
% opti_p.subject_to(); %Remove all the constraints
% opti_p.subject_to(const_p);
% opti_p.minimize(substitute(total_cost,yCPs,yCPs_par)); 
% my_func_p = opti_p.to_function('my_func_p', vars, results_vars, names, results_names);
% 
% %CREATE SOLVER FOR YAW, POSITION IS FIXED
% opti_y=opti.copy;
% opti_y.subject_to(); %Remove all the constraints
% opti_y.subject_to(const_y);
% opti_y.minimize(substitute(total_cost,pCPs,pCPs_par)); 
% my_func_y = opti_y.to_function('my_func_y', vars, results_vars, names, results_names);
% 
% % sol=my_func_p( names_value{:}); %Dummy call to avoid the overhead of the first iteration (not sure why it happens)
% % sol=my_func_y( names_value{:}); %Dummy call to avoid the overhead of the first iteration (not sure why it happens)
%   
%  
% all_costs=[];
% ms_p=[];
% ms_y=[];
% inner_iterations_p={};
% inner_iterations_y={};
% num_outer_it=10;
% for i=1:num_outer_it 
%     disp('==============================================')
%     
%     tic()
%     sol=my_func_p( names_value{:});     ms_p=[ms_p 1000*toc()];
% 
%     tmp_p=get_stats(my_func_p);
%     tmp_p.iterations.obj
%     inner_iterations_p{end+1}=tmp_p.iterations.obj;
% 
%     names_value{2}= sol.pCPs;%Update Pos
%     names_value{6}= sol.pCPs;%Update Pos par
% 
%     tic()
%     sol=my_func_y( names_value{:});
%     ms_y=[ms_y 1000*toc()];
%     
%     tmp_y=get_stats(my_func_y);
%     inner_iterations_y{end+1}=tmp_y.iterations.obj;
% 
%     
%     names_value{4}= sol.yCPs;%Update Pos
%     names_value{8}= sol.yCPs;%Update Pos
% 
%     all_costs=[all_costs full(sol.total_cost)];
% 
% end
% 
% toc();
% 
% 
% tic();
% sol_single=my_func( names_value{:} );
% ms_single=1000*toc();
% 
% close all;
% figure; hold on;
% yline(full(sol_single.total_cost),'--')
% 
% for i=1:num_outer_it
%     plot(i*ones(size(inner_iterations_p{i})), inner_iterations_p{i},'-b')
%     scatter(i*ones(size(inner_iterations_p{i})), inner_iterations_p{i},'b','filled')
% 
%     plot(i*ones(size(inner_iterations_y{i})), inner_iterations_y{i},'-r')
%     scatter(i*ones(size(inner_iterations_y{i})), inner_iterations_y{i},'r','filled')
%     
%     if(i<num_outer_it)
%         plot([i,i+1],[inner_iterations_y{i}(end),inner_iterations_p{i+1}(1)], '--m')
%     end
% end
% 
% plot([0,1],[inner_iterations_p{1}(1),inner_iterations_p{1}(1)], '--m')
% 
%  
% xlabel('Outer iteration'); ylabel('Cost');
% 
% figure; hold on; plot(ms_p); plot(ms_y); yline(ms_single,'--')
% legend('opt. pos','opt. yaw','joint'); xlabel('Outer iteration'); ylabel('time (ms)');
% 
%                                           
% sol=my_func_p( names_value{:});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% [end of block coordinate descent]

%% Visualization of the hessians
% figure;spy(hessian(opti.f,opti.x),15,'sk'); set(get(gca,'Children'),'MarkerFaceColor','b')
% exportAsPdf(gcf,'hessian_coupled');
% figure;spy(hessian(opti_y.f,opti_y.x),22,'sr'); set(get(gca,'Children'),'MarkerFaceColor','r')
% figure;spy(hessian(opti_p.f,opti_p.x),22,'sr'); set(get(gca,'Children'),'MarkerFaceColor','r')


 
%%
% statistics=get_stats(my_function); %See functions defined below
% full(sol.pCPs)
% full(sol.yCPs)

% function [const_p,const_y]=addDynLimConstraints(const_p,const_y, sp, sy, basis, v_max_scaled, a_max_scaled, j_max_scaled, ydot_max_scaled)
% 
%     const_p=[const_p sp.getMaxVelConstraints(basis, v_max_scaled)];      %Max vel constraints (position)
%     const_p=[const_p sp.getMaxAccelConstraints(basis, a_max_scaled)];    %Max accel constraints (position)
%     const_p=[const_p sp.getMaxJerkConstraints(basis, j_max_scaled)];     %Max jerk constraints (position)
%     const_y=[const_y sy.getMaxVelConstraints(basis, ydot_max_scaled)];   %Max vel constraints (yaw)
% 
% end

%% Functions

function result=substituteWithSolution(expression, all_var_solved, all_params_and_init_guesses)

  import casadi.*
    result=zeros(size(expression));
    for i=1:size(expression,1)
        for j=1:size(expression,2)
            
                tmp=expression(i,j);
                
                %Substitute first the solution
                for ii=1:numel(all_var_solved)
                    tmp=substitute(tmp,all_var_solved{ii}.param, all_var_solved{ii}.value);
                end
                
                 %And then substitute the parameters
                for ii=1:numel(all_params_and_init_guesses)
                    tmp=substitute(tmp,all_params_and_init_guesses{ii}.param, all_params_and_init_guesses{ii}.value);
                end
            
            result(i,j)=convertMX2Matlab(tmp);
        end
    end


end


function [t_proc_total, t_wall_total]= timeInfo(my_func)
    tmp=get_stats(my_func);
    
    %%See https://github.com/casadi/casadi/wiki/FAQ:-is-the-bottleneck-of-my-optimization-in-CasADi-function-evaluations-or-in-the-solver%3F
    
    %This is the time spent evaluating the functions
    t_casadi=  (tmp.t_wall_nlp_f+tmp.t_wall_nlp_g+tmp.t_wall_nlp_grad+tmp.t_wall_nlp_grad_f+tmp.t_wall_nlp_hess_l+tmp.t_wall_nlp_jac_g);
    
    t_ipopt= tmp.t_wall_total-t_casadi;

    cprintf('key','Total time=%.2f ms [Casadi=%.2f%%, IPOPT=%.2f%%]\n',tmp.t_wall_total*1000, 100*t_casadi/tmp.t_wall_total, 100*t_ipopt/tmp.t_wall_total    )
    
    
    
    t_wall_total=tmp.t_wall_total;
    t_proc_total=tmp.t_proc_total;    
    
end

%Taken from https://gist.github.com/jgillis/9d12df1994b6fea08eddd0a3f0b0737f
%See discussion at https://groups.google.com/g/casadi-users/c/1061E0eVAXM/m/dFHpw1CQBgAJ
function [stats] = get_stats(f)
  dep = 0;
  % Loop over the algorithm
  for k=0:f.n_instructions()-1
%      fprintf("Trying with k= %d\n", k)
    if f.instruction_id(k)==casadi.OP_CALL
      fprintf("Found k= %d\n", k)
      d = f.instruction_MX(k).which_function();
      if d.name()=='solver'
        my_file=fopen('./casadi_generated_files/index_instruction.txt','w'); %Overwrite content
        fprintf(my_file,'%d\n',k);
        dep = d;
        break
      end
    end
  end
  if dep==0
    stats = struct;
  else
    stats = dep.stats(1);
  end
end

function a=createStruct(name,param,value)
    a.name=name;
    a.param=param;
    a.value=value;
end

function result=mySig(gamma,x)
    result=(1/(1+exp(-gamma*x)));
end

function result=getUSimpsonJ(span_interval, sp,j, t_simpson)
    t_init_interval=min(span_interval);   
    t_final_interval=max(span_interval);
    delta_interval=t_final_interval-t_init_interval;
    
    tsf=t_simpson; %tsf is a filtered version of  t_simpson
    tsf=tsf(tsf>=min(t_init_interval));
    if(j==(sp.num_seg))
        tsf=tsf(tsf<=max(t_final_interval));
    else
        tsf=tsf(tsf<max(t_final_interval));
    end
    result=(tsf-t_init_interval)/delta_interval;
end
% if(strcmp(linear_solver_name,'ma57'))
%    opts.ipopt.ma57_automatic_scaling='no';
% end
%opts.ipopt.hessian_approximation = 'limited-memory';
% jit_compilation=false; %If true, when I call solve(), Matlab will automatically generate a .c file, convert it to a .mex and then solve the problem using that compiled code
% opts.jit=jit_compilation;
% opts.compiler='clang';
% opts.jit_options.flags='-O0';  %Takes ~15 seconds to generate if O0 (much more if O1,...,O3)
% opts.jit_options.verbose=true;  %See example in shallow_water.cpp
% opts.enable_forward=false; %Seems this option doesn't have effect?
% opts.enable_reverse=false;
% opts.enable_jacobian=false;
% opts.qpsol ='qrqp';  %Other solver
% opti.solver('sqpmethod',opts);