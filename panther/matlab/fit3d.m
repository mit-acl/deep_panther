close all; clc;clear;
doSetup();
import casadi.*


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%    FUNCTION TO FIT A SPLINE TO POSITION SAMPLES     %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%all_pos should be sampled uniformly, including first and last point
%The total number of samples is num_samples

%If you find the error "evaluation failed" --> increase num_samples or
%reduce deg_pos or num_seg

deg_pos=3;
dim_pos=3;
num_seg=3;
opti = casadi.Opti();

num_samples=15;

t0_n=0;
tf_n=1;

sp=MyClampedUniformSpline(t0_n,tf_n, deg_pos, dim_pos, num_seg, opti); 

all_pos=MX.sym('all_pos',dim_pos,num_samples);
cost_function=0;
i=1;
for ti=linspace(t0_n, tf_n, num_samples)
    
    dist=(sp.getPosT(ti)-all_pos(:,i));
    cost_function = cost_function + dist'*dist; 
    
    i=i+1;
    
end


lagrangian = cost_function;

variables=[sp.getCPsAsMatrix() ]; 

kkt_eqs=jacobian(lagrangian, variables)'; %I want kkt=[0 0 ... 0]'

%Obtain A and b
b=-casadi.substitute(kkt_eqs, variables, zeros(size(variables))); %Note the - sign
A=jacobian(kkt_eqs, variables);

solution=A\b;  %Solve the system of equations

f= Function('f', {all_pos }, {reshape(solution(1:end), dim_pos,-1)}, ...
                 {'all_pos'}, {'result'} );
% f=f.expand();

t=linspace(0, 2, num_samples);

all_pos_value=[sin(t)+2*sin(2*t);
               cos(t)-2*cos(2*t);
               -sin(3*t)];


solution=f(all_pos_value);

cost_function=substitute(cost_function, sp.getCPsAsMatrix, full(solution));
cost_function=substitute(cost_function, all_pos, all_pos_value);
convertMX2Matlab(cost_function)

% sp_tmp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti);  %creating another object to not mess up with sy
sp.updateCPsWithSolution(full(solution));


% sp_tmp.plotPosVelAccelJerk();

sp.plotPos3D();
scatter3(all_pos_value(1,:), all_pos_value(2,:), all_pos_value(3,:))

f.save('./casadi_generated_files/fit3d.casadi') 

% solution=convertMX2Matlab(A)\convertMX2Matlab(b);  %Solve the system of equations
% sy.updateCPsWithSolution(solution(1:end-3)');