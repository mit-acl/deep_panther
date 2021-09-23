% This files implements block-coordinate descent on yaw and pos*/

close all; clc;clear;

doSetup();

import casadi.*
opti = casadi.Opti();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% CONSTANTS! %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
t0=0;   tf=10.5;

dim_pos=3;  dim_yaw=1;

offset_vel=0.1;

assert(tf>t0);

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
% c_costs.dist_im_cost=         opti.parameter(1,1);

Ra=opti.parameter(1,1);

thetax_FOV_deg=opti.parameter(1,1);    %total angle of the FOV in the x direction
thetay_FOV_deg=opti.parameter(1,1);    %total angle of the FOV in the y direction

thetax_half_FOV_deg=thetax_FOV_deg/2.0; %half of the angle of the cone
thetax_half_FOV_rad=thetax_half_FOV_deg*pi/180.0;

thetay_half_FOV_deg=thetay_FOV_deg/2.0; %half of the angle of the cone
thetay_half_FOV_rad=thetay_half_FOV_deg*pi/180.0;

%%%%% Initial and final conditions
p0=opti.parameter(3,1); v0=opti.parameter(3,1); a0=opti.parameter(3,1);
pf=opti.parameter(3,1); vf=opti.parameter(3,1); af=opti.parameter(3,1);
y0=opti.parameter(1,1); ydot0=opti.parameter(1,1); 
yf=opti.parameter(1,1); ydotf=opti.parameter(1,1);

%%%%% Planes
n={}; d={};
for i=1:(num_max_of_obst*num_seg)
    n{i}=opti.parameter(3,1); 
    d{i}=opti.parameter(1,1);
end

%%% Maximum velocity and acceleration
v_max=opti.parameter(3,1);
a_max=opti.parameter(3,1);
j_max=opti.parameter(3,1);
ydot_max=opti.parameter(1,1);

total_time=opti.parameter(1,1); %This allows a different t0 and tf than the one above 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% CREATION OF THE SPLINES! %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sp=MyClampedUniformSpline(t0,tf,deg_pos, dim_pos, num_seg, opti); %spline position.
sy=MyClampedUniformSpline(t0,tf,deg_yaw, dim_yaw, num_seg, opti); %spline yaw.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% CONSTRAINTS! %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

scaling=(tf-t0)/total_time;

v0_scaled=v0/scaling;
a0_scaled=a0/(scaling^2);
ydot0_scaled=ydot0/scaling;


vf_scaled=vf/scaling;
af_scaled=af/(scaling^2);
ydotf_scaled=ydotf/scaling;

v_max_scaled=v_max/scaling;
a_max_scaled=a_max/(scaling^2); 
j_max_scaled=j_max/(scaling^3);

ydot_max_scaled=ydot_max/scaling; %v_max for yaw

%Initial conditions
const_p{end+1}=sp.getPosT(t0)== p0;
const_p{end+1}=sp.getVelT(t0)== v0_scaled;
const_p{end+1}=sp.getAccelT(t0)== a0_scaled;
const_y{end+1}=sy.getPosT(t0)== y0;
const_y{end+1}=sy.getVelT(t0)== ydot0_scaled ;

%Final conditions
% opti.subject_to( sp.getPosT(tf)== pf );
const_p{end+1}=sp.getVelT(tf)== vf_scaled;
const_p{end+1}=sp.getAccelT(tf)== af_scaled;
const_y{end+1}=sy.getVelT(tf)==ydotf_scaled; % Needed: if not (and if you are minimizing ddyaw), dyaw=cte --> yaw will explode


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



[const_p,const_y]=addDynLimConstraints(const_p,const_y, sp, sy, basis, v_max_scaled, a_max_scaled, j_max_scaled, ydot_max_scaled);


g=9.81;
%Compute perception cost
dist_im_cost=0;
vel_im_cost=0;
fov_cost=0;

clear i
t_simpson=linspace(t0,tf,num_samples_simpson);
delta_simpson=(t_simpson(2)-t_simpson(1));



u=MX.sym('u',1,1); %it must be defined outside the loop (so that then I can use substitute it regardless of the interval
w_fevar=MX.sym('w_fevar',3,1); %it must be defined outside the loop (so that then I can use substitute it regardless of the interval
w_velfewrtworldvar=MX.sym('w_velfewrtworld',3,1);
yaw= MX.sym('yaw',1,1);  
simpson_index=1;
simpson_coeffs=[];

all_target_isInFOV=[];

f=0.05;%focal length in meters

for j=1:sp.num_seg
    

    w_p = sp.getPosU(u,j);
    a=sp.getAccelU(u,j)+[0;0;g];
    
    %%%%
    a1=a(1); a2=a(2); a3=a(3);
    na2= a1*a1 + a2*a2 + a3*a3;
    na=sqrt(na2);    
    na2_plus_a3na=na2 + a(3)*na;
    
    yaw=sy.getPosU(u,j);
    
    b1= [ (1-(a1^2)/na2_plus_a3na)*cos(yaw) -    a1*a2*sin(yaw)/na2_plus_a3na;
        
          -a1*a2*cos(yaw)/na2_plus_a3na     +    (1 - (a2^2)/na2_plus_a3na)*sin(yaw);
          
           (a1/na)*cos(yaw)                 +    (a2/na)*sin(yaw)         ]; %note that, by construction, norm(b1)=1;
    
    w_e=-w_p; %Asumming feature is in [0 0 0]';
    
    f=cos(thetax_half_FOV_deg*pi/180.0)*norm(w_e) - b1'*w_e; %Constraint is f<=0
    
    fov_cost_j=max(0,f)^3; %Penalty associated with the constraint
    

%     isInFOV=-cos(thetax_half_FOV_deg*pi/180.0) + b1'*w_e/norm(w_e);%This has to be >=0
%     
%     fov_cost_j=-isInFOV; Note that, if I use this, and isInFOV is
%     -cos(XX)*norm(w_e) + b1'*w_e, then this part of the cost is unbounded
%     (I can always keep decressing this term of the cost by modifying
%     norm(w_e). As soon as other terms are added to the cost, this is
%     fixed.
    
    %%%%%%%%%%%%%%%%%%
      
    span_interval=sp.timeSpanOfInterval(j);
    
    u_simpson{j}=getUSimpsonJ(span_interval, sp,j, t_simpson);    
    
    for u_i=u_simpson{j}
                
        simpson_coeff=getSimpsonCoeff(simpson_index,num_samples_simpson);
        fov_cost=fov_cost + (delta_simpson/3.0)*simpson_coeff*substitute( fov_cost_j,u,u_i); 
        
        simpson_index=simpson_index+1;
        
    end
end

%Cost
pos_smooth_cost=sp.getControlCost();
yaw_smooth_cost=sy.getControlCost();

final_pos_cost=(sp.getPosT(tf)- pf)'*(sp.getPosT(tf)- pf);
final_yaw_cost=(sy.getPosT(tf)- yf)^2;

total_cost=c_pos_smooth*pos_smooth_cost+...
           c_yaw_smooth*yaw_smooth_cost+... 
           c_fov*fov_cost+...
           c_final_pos*final_pos_cost+...
           c_final_yaw*final_yaw_cost;

opti.minimize(simplify(total_cost));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLVE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

all_nd=[];
for i=1:(num_max_of_obst*num_seg)
    all_nd=[all_nd [n{i};d{i}]];
end

pCPs=sp.getCPsAsMatrix();
yCPs=sy.getCPsAsMatrix();


yCPs_par=opti.parameter(1,size(yCPs,2));
pCPs_par=opti.parameter(3,size(pCPs,2));


v_max_value=1.6*ones(3,1);
a_max_value=50*ones(3,1);
j_max_value=50*ones(3,1);
ydot_max_value=1.0;

total_time_value=10.5;
thetax_FOV_deg_value=80;
thetay_FOV_deg_value=80;
Ra_value=12.0;
y0_value=0.0;
yf_value=0.0;
ydot0_value=0.0;
ydotf_value=0.0;

p0_value=[-4;0.0;0.0];
v0_value=[0;0;0];
a0_value=[0;0;0];

pf_value=[4.0;0.0;0.0];
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


all_params= [ {createStruct('thetax_FOV_deg', thetax_FOV_deg, thetax_FOV_deg_value)},...
              {createStruct('thetay_FOV_deg', thetay_FOV_deg, thetay_FOV_deg_value)},...
              {createStruct('Ra', Ra, Ra_value)},...
              {createStruct('p0', p0, p0_value)},...
              {createStruct('v0', v0, v0_value)},...
              {createStruct('a0', a0, a0_value)},...
              {createStruct('pf', pf, pf_value)},...
              {createStruct('vf', vf, vf_value)},...
              {createStruct('af', af, af_value)},...
              {createStruct('y0', y0, y0_value)},...
              {createStruct('ydot0', ydot0, ydot0_value)},...
              {createStruct('yf', yf, yf_value)},...
              {createStruct('ydotf', ydotf, ydotf_value)},...
              {createStruct('v_max', v_max, v_max_value)},...
              {createStruct('a_max', a_max, a_max_value)},...
              {createStruct('j_max', j_max, j_max_value)},...
              {createStruct('ydot_max', ydot_max, ydot_max_value)},... 
              {createStruct('total_time', total_time, total_time_value)},...
              {createStruct('all_nd', all_nd, all_nd_value)},...
              {createStruct('c_pos_smooth', c_pos_smooth, 10.0)},...
              {createStruct('c_yaw_smooth', c_yaw_smooth, 0.0)},...
              {createStruct('c_fov', c_fov, 1.0)},...
              {createStruct('c_final_pos', c_final_pos, 10.0)},...
              {createStruct('c_final_yaw', c_final_yaw, 0.0)}];


tmp1=[   -4.0000   -4.0000   -4.0000    0.7111  1.0 2.0   4         4          4;
         0         0         0           0.5    1.0    0.5  0           0         0;
         0         0         0             0       0   0    0           0         0];
     
tmp2=[   -0.0000   -0.0000    0.2754  1.0 1.5  2.1131    2.6791    2.6791];

all_params_and_init_guesses=[{createStruct('pCPs', pCPs, tmp1)},...
                             {createStruct('yCPs', yCPs, tmp2)},...
                             {createStruct('pCPs_par', pCPs_par, tmp1)},...
                             {createStruct('yCPs_par', yCPs_par, tmp2)},...
                             all_params];

vars=[];
names=[];
for i=1:numel(all_params_and_init_guesses)
    vars=[vars {all_params_and_init_guesses{i}.param}];
    names=[names {all_params_and_init_guesses{i}.name}];
end

names_value={};
values=[];
for i=1:numel(all_params_and_init_guesses)
    names_value{end+1}=all_params_and_init_guesses{i}.name;
    names_value{end+1}=double2DM(all_params_and_init_guesses{i}.value); 
end


opts = struct;
opts.expand=true; %When this option is true, it goes WAY faster!
opts.print_time=true;
opts.ipopt.print_level=print_level; 
%opts.ipopt.print_frequency_iter=1e10;%1e10 %Big if you don't want to print all the iteratons
opts.ipopt.linear_solver=linear_solver_name;
opti.solver('ipopt',opts); %{"ipopt.hessian_approximation":"limited-memory"} 


opti.subject_to([const_p, const_y])

results_vars={pCPs,yCPs, total_cost,pos_smooth_cost, yaw_smooth_cost, fov_cost, final_pos_cost, final_yaw_cost};
results_names={'pCPs','yCPs','total_cost','pos_smooth_cost','yaw_smooth_cost','fov_cost','final_pos_cost','final_yaw_cost'};

my_func = opti.to_function('my_func', vars, results_vars, names, results_names);
sol=my_func( names_value{:});
full(sol.pCPs)
full(sol.yCPs)
%%
import casadi.*
clc
%CREATE SOLVER FOR POSITION, YAW IS FIXED
opti_p=opti.copy;

 
%See https://github.com/casadi/casadi/wiki/FAQ:-how-to-perform-jit-for-function-evaluations-of-my-optimization-problem%3F
opts.jit=true;%If true, when I call solve(), Matlab will automatically generate a .c file, convert it to a .mex and then solve the problem using that compiled code
opts.compiler='shell';
opts.jit_options.flags='-O3';  %Takes ~15 seconds to generate if O0 (much more if O1,...,O3)
opts.jit_options.verbose=true;  %See example in shallow_water.cpp
opti.solver('ipopt',opts); 

opti_p.subject_to(); %Remove all the constraints
opti_p.subject_to(const_p);
opti_p.minimize(substitute(total_cost,yCPs,yCPs_par)); 
my_func_p = opti_p.to_function('my_func_p', vars, results_vars, names, results_names);

%CREATE SOLVER FOR YAW, POSITION IS FIXED
opti_y=opti.copy;
opti_y.subject_to(); %Remove all the constraints
opti_y.subject_to(const_y);
opti_y.minimize(substitute(total_cost,pCPs,pCPs_par)); 
my_func_y = opti_y.to_function('my_func_y', vars, results_vars, names, results_names);
%%



%%
% 
% sol=my_func_p( names_value{:}); %Dummy call to avoid the overhead of the first iteration (not sure why it happens)
% sol=my_func_y( names_value{:}); %Dummy call to avoid the overhead of the first iteration (not sure why it happens)
% 
% sol=my_func_p( names_value{:}); %Dummy call to avoid the overhead of the first iteration (not sure why it happens)
% sol=my_func_y( names_value{:}); %Dummy call to avoid the overhead of the first iteration (not sure why it happens)
% 
% sol=my_func_p( names_value{:}); %Dummy call to avoid the overhead of the first iteration (not sure why it happens)
% sol=my_func_y( names_value{:}); %Dummy call to avoid the overhead of the first iteration (not sure why it happens)
%  
 
all_costs=[];
ms_p=[];
ms_y=[];
inner_iterations_p={};
inner_iterations_y={};
num_outer_it=10;
for i=1:num_outer_it
    
    disp('==============================================')
    
    tic()
    sol=my_func_p( names_value{:});     ms_p=[ms_p 1000*toc()];
%     if(i==1)
    tmp_p=get_stats(my_func_p);
    tmp_p.iterations.obj
    inner_iterations_p{end+1}=tmp_p.iterations.obj;
%     end
    names_value{2}= sol.pCPs;%Update Pos
    names_value{6}= sol.pCPs;%Update Pos par

    % all_costs=[all_costs full(sol.cost)];

    tic()
    sol=my_func_y( names_value{:});
    ms_y=[ms_y 1000*toc()];
    
    tmp_y=get_stats(my_func_y);
    inner_iterations_y{end+1}=tmp_y.iterations.obj;

    
    names_value{4}= sol.yCPs;%Update Pos
    names_value{8}= sol.yCPs;%Update Pos

    all_costs=[all_costs full(sol.total_cost)];

end

%%

toc();


tic();
sol_single=my_func( names_value{:} );
ms_single=1000*toc();

close all;
figure; hold on;
yline(full(sol_single.total_cost),'--')

for i=1:num_outer_it
    plot(i*ones(size(inner_iterations_p{i})), inner_iterations_p{i},'-b')
    scatter(i*ones(size(inner_iterations_p{i})), inner_iterations_p{i},'b','filled')

    plot(i*ones(size(inner_iterations_y{i})), inner_iterations_y{i},'-r')
    scatter(i*ones(size(inner_iterations_y{i})), inner_iterations_y{i},'r','filled')
    
    if(i<num_outer_it)
        plot([i,i+1],[inner_iterations_y{i}(end),inner_iterations_p{i+1}(1)], '--m')
    end
end

plot([0,1],[inner_iterations_p{1}(1),inner_iterations_p{1}(1)], '--m')

% full(sol_single.pCPs)
% full(sol_single.yCPs)


% plot(all_costs); 
xlabel('Outer iteration'); ylabel('Cost');

figure;
plot(ms_p); hold on
plot(ms_y); xlabel('Outer iteration'); ylabel('time (ms)');
yline(ms_single,'--')
legend('opt. pos','opt. yaw','joint')

                                          
sol=my_func_p( names_value{:});


%%

%Store solution
sp.updateCPsWithSolution(full(sol.pCPs))
sy.updateCPsWithSolution(full(sol.yCPs))

%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOTTING! %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sp.plotPosVelAccelJerk(v_max_value, a_max_value, j_max_value)
% % sp.plotPosVelAccelJerkFiniteDifferences();
% sy.plotPosVelAccelJerk(ydot_max_value)
% sy.plotPosVelAccelJerkFiniteDifferences();

sp.plotPos3D();
plotSphere( sp.getPosT(t0),0.2,'b'); plotSphere( sp.getPosT(tf),0.2,'r'); 

view([280,15]); axis equal
% 
disp("Plotting")
for t_i=t_simpson %t0:0.3:tf  
    
    w_t_b = sp.getPosT(t_i);
    accel = sp.getAccelT(t_i);% sol.value(A{n})*Tau_i;
    yaw = sy.getPosT(t_i);
%         psiT=sol.value(Psi{n})*Tau_i;

    qabc=qabcFromAccel(accel, 9.81);

    qpsi=[cos(yaw/2), 0, 0, sin(yaw/2)]; %Note that qpsi has norm=1
    q=multquat(qabc,qpsi); %Note that q is guaranteed to have norm=1

    w_R_b=toRotMat(q);
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

grid on; xlabel('x'); ylabel('y'); zlabel('z'); 
camlight
lightangle(gca,45,0)


syms x y z real
for i=1:size(all_nd_value,2)
   fimplicit3(all_nd_value(:,i)'*[x;y;z;1],[-4 4 -4 4 -2 2], 'MeshDensity',2, 'FaceAlpha',0.6) 
end

view(-91,90)



%% Visualization of the hessians
figure;spy(hessian(opti.f,opti.x),15,'sk'); set(get(gca,'Children'),'MarkerFaceColor','b')
% exportAsPdf(gcf,'hessian_coupled');
% figure;spy(hessian(opti_y.f,opti_y.x),22,'sr'); set(get(gca,'Children'),'MarkerFaceColor','r')
% figure;spy(hessian(opti_p.f,opti_p.x),22,'sr'); set(get(gca,'Children'),'MarkerFaceColor','r')


 
%%
% statistics=get_stats(my_function); %See functions defined below
% full(sol.pCPs)
% full(sol.yCPs)

function [const_p,const_y]=addDynLimConstraints(const_p,const_y, sp, sy, basis, v_max_scaled, a_max_scaled, j_max_scaled, ydot_max_scaled)

    const_p=[const_p sp.getMaxVelConstraints(basis, v_max_scaled)];      %Max vel constraints (position)
    const_p=[const_p sp.getMaxAccelConstraints(basis, a_max_scaled)];    %Max accel constraints (position)
    const_p=[const_p sp.getMaxJerkConstraints(basis, j_max_scaled)];     %Max jerk constraints (position)
    const_y=[const_y sy.getMaxVelConstraints(basis, ydot_max_scaled)];   %Max vel constraints (yaw)

end

%% Functions

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