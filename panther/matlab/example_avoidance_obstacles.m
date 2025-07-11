close all; clc;clear;
doSetup();
import casadi.*


waypoints=[[40.          4.          1.         30.06858282  3.          0. 1.        ];
[ 51.          23.           2.15058651 128.65980825  -0.44914504  1.           1.        ];
[66.         11.          2.         81.46923439 -2.83078838  1. 1.        ];
[86.         14.          1.         21.25050551  2.96401609  1.  1.        ];
[ 93.          32.           2.         -63.43494882  -2.56063897  0.           1.        ];
[ 73.          42.           1.         -64.98310652   3.  1.           1.        ];
[58.         49.          1.86750311 77.27564431  2.0408695   0. 1.        ];
[89.         56.          3.         20.22485943 -2.8273397   0. 1.        ];
[ 96.          75.           2.         -23.49856568   2.28332282  0.           1.        ];
[  86.           98.            3.         -113.96248897    2.90625743   0.            1.        ];
[  68.           90.            4.         -147.99461679    0.   0.            1.        ];
[ 58.          74.           4.         -39.47245985  -3.  1.           1.        ];
[  44.           91.            2.84583838 -153.43494882   -1.96952379   0.            1.        ];
[ 33.          69.           2.         -50.19442891   2.44384119 0.           1.        ];
[ 15.          84.           3.         176.82016988  -3.  1.           1.        ];
[  16.           66.            2.05520532 -165.37912601   -2.54185243   1.            1.        ];
[ 10.          43.           1.         130.23635831   3.  1.           1.        ];
[  23.           32.            1.89247232 -125.83765295   -2.30176089   0.            1.        ];
[  5.          19.           1.         153.43494882   3.  1.           1.        ];
[ 13.           3.           1.93749886 173.43494882   0.  0.           1.        ]];
%%


%Solve without obstacles
disp("GOING TO OBTAIN INITIAL GUESS...")
cps_sol_initial_guess=getSolution(false,waypoints,[],false);

%Solve with obstacles
disp("GOING TO OBTAIN OPTIMAL SOLUTION...")
cps_sol=getSolution(true,waypoints,cps_sol_initial_guess,true);

function cps_sol=getSolution(do_plots,waypoints, initial_guess,impose_obstacles)

if(impose_obstacles==false)
    waypoints(:,6)=0;
    %Modify the waypoints of the obstacles such that they don't pass
    waypoints(:,3) = waypoints(:,3)+7*((waypoints(:,6)==1));
end

opti = casadi.Opti();

num_max_of_obst = 9;

t0_n=0.0; 
tf_n=1.0;
total_time=300;
total_time_n=(tf_n-t0_n);
alpha=total_time/total_time_n;  %Please read explanation_normalization.svg

Ra=5.0; %Radius of the obstacles

dim_pos=3;
deg_pos=3;
num_seg_between_two_wp = 3;
basis="MINVO";

num_waypoints=size(waypoints,1);
num_seg=(num_waypoints-1)*num_seg_between_two_wp;

sp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.

constraints=[];

%Initial heading
constraints{end+1}= sp.getVelT(t0_n)== [1 1 0]' ;

%Final heading
constraints{end+1}= sp.getVelT(tf_n)== [-1 -1 0]' ;

%Dynamic limits
v_max=20*ones(1,3);     a_max=200*ones(1,3);        j_max=50*ones(1,3);
%Dynamic limits normalized
v_max_n=v_max*alpha;   a_max_n=a_max*(alpha^2);  j_max_n=j_max*(alpha^3);

constraints=[constraints sp.getMaxVelConstraints(basis, v_max_n)];      %Max vel constraints (position)
constraints=[constraints sp.getMaxAccelConstraints(basis, a_max_n)];    %Max accel constraints (position)
%constraints=[constraints sp.getMaxJerkConstraints(basis, j_max_n)];     %Max jerk constraints (position)

opts = struct;
opts.expand=true; %When this option is true, it goes WAY faster!
opts.print_time=0;
opts.ipopt.print_level=2; 
opts.ipopt.max_iter=1000;
opti.solver('ipopt',opts); %{"ipopt.hessian_approximation":"limited-memory"} 

%Force it to pass through the non-obstacles waypoints
penalty_wp_obstacles=0;
for i=1:num_waypoints
    tn_passing_wp_i=(i-1)*tf_n/(num_waypoints-1);
    pos_uav=sp.getPosT(tn_passing_wp_i);
    pos_wp=waypoints(i,1:3)';
    if(waypoints(i,6)==0) %If it's not an obstacle
        constraints{end+1} = pos_uav == pos_wp;
    else %If it's not an obstacle
        distance_squared=(pos_uav-pos_wp)'*(pos_uav-pos_wp);
        penalty_wp_obstacles = penalty_wp_obstacles + distance_squared;
    end

end


%Force it to avoid the obstacles
for i=1:num_waypoints
    if(waypoints(i,6)==0) %If it's not an obstacle
        continue;
    end
    tn_passing_wp_i=(i-1)*tf_n/(num_waypoints-1);

    pos_wp=waypoints(i,1:3)';
    for tn=max((tn_passing_wp_i-0.2),0):0.005:min((tn_passing_wp_i+0.2),1)
        pos_uav=sp.getPosT(tn);
        distance_squared=(pos_uav-pos_wp)'*(pos_uav-pos_wp);
        constraints{end+1} = distance_squared>=(Ra*Ra);
    end
end

%INITIAL GUESS
if(numel(initial_guess)>0) %If initial guess provided
    opti.set_initial(sp.getCPsAsMatrix(),initial_guess)
end

%SOLVE
opti.subject_to(constraints)
opti.minimize( penalty_wp_obstacles + (1e-10)*sp.getControlCost())% 

tic
sol = opti.solve();
toc

cps_sol=sol.value(sp.getCPsAsMatrix());

%GET SOLUTION
sp.updateCPsWithSolution(cps_sol)

if(do_plots)
    disp("MAKING PLOTS...")

    nonobsts_wp=waypoints(waypoints(:, 6) == 0, :); %keep the ones that have 0
    obsts_wp=waypoints(waypoints(:, 6) == 1, :); %keep the ones that have 1
    pos_obsts_wp=obsts_wp(:,1:3)';
    pos_nonobsts_wp=nonobsts_wp(:,1:3)';

    %PLOTTING
    sp.plotPosVelAccelJerk(v_max_n, a_max_n, j_max_n)
    sp.plotPos3D()
    lighting gouraud; shading interp; light('Position', [1 0 1], 'Style', 'infinite');
    for obst_i=1:size(pos_obsts_wp,2)
       plotSphere(pos_obsts_wp(:,obst_i), Ra, 'r');
    end
    for obst_i=1:size(pos_nonobsts_wp,2)
       plotSphere(pos_nonobsts_wp(:,obst_i), 2, 'b');
    end
    set(gcf, 'Alpha', 0.01);
    xlim([min(waypoints(:,1))-2*Ra,max(waypoints(:,1))+2*Ra])
    ylim([min(waypoints(:,2))-2*Ra,max(waypoints(:,2))+2*Ra])
    zlim([min(waypoints(:,3))-2*Ra,max(waypoints(:,3))+2*Ra])
end

end


%%CURVATURE CONSTRAINTS
% Do not work well with standard B-Splines
% Better impose max accel, jerk,...
% for tn=linspace(t0_n+0.1,tf_n-0.1,2)
%     vel=sp.getVelT(tn);
%     accel=sp.getAccelT(tn);
%     xp=vel(1); yp=vel(2); zp=vel(3);
%     xpp=accel(1); ypp=accel(2); zpp=accel(3);
%     kappa = sqrt( (zpp*yp-ypp*zp)^2 + (xpp*zp-zpp*xp)^2  + (ypp*xp-xpp*yp)^2)/((xp^2 +yp^2 +zp^2)^(3/2));
%     constraints{end+1} = kappa<=1/0.5;
% end