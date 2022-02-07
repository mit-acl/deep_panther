% /* ----------------------------------------------------------------------------
%  * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
%  * Massachusetts Institute of Technology
%  * All Rights Reserved
%  * Authors: Jesus Tordesillas, et al.
%  * See LICENSE file for the license information
%  * -------------------------------------------------------------------------- */

close all; clc;clear;  doSetup(); import casadi.*

const_p={};    const_y={};
opti = casadi.Opti();
basis="MINVO";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t0_n=0.0;
tf_n=1.0;
deg_pos=3;
dim_pos=3;
num_seg =4; %number of segments

sp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.

ncpts=numel(sp.CPoints);
variables=sym(zeros(dim_pos,ncpts));
coord=['x','y','z'];
for i=1:dim_pos
    for j=1:ncpts
        variables(i,j)=sym("q"+num2str(j)+coord(i),'real');
    end
end
% variables=sym('q%d%d',[dim_pos, numel(sp.CPoints)]);

sp.updateCPsWithSolution(variables)

%%%%% Initial and final conditions, and max values
%FOR POSITION
% p0=opti.parameter(3,1); v0=opti.parameter(3,1); a0=opti.parameter(3,1);
% pf=opti.parameter(3,1); vf=opti.parameter(3,1); af=opti.parameter(3,1);



v_max=ones(3,1);%getSym3DVector("v_max");
a_max=ones(3,1);%getSym3DVector("a_max");
j_max=ones(3,1);%getSym3DVector("j_max");
alpha=sym('alpha','real');

p0=zeros(3,1);%getSym3DVector("p0");
v0=getSym3DVector("v0");
a0=getSym3DVector("a0");

% pf=getSym3DVector("pf");
vf=zeros(3,1);%getSym3DVector("vf");
af=zeros(3,1);%getSym3DVector("af");

%Normalized v0, a0, v_max,...
v0_n=v0*alpha;
a0_n=a0*(alpha^2);
vf_n=vf*alpha;
af_n=af*(alpha^2);
v_max_n=v_max*alpha;
a_max_n=a_max*(alpha^2); 
j_max_n=j_max*(alpha^3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Initial conditions
% const_p{end+1}= sp.getPosT(t0_n)== p0 ;
% const_p{end+1}= sp.getVelT(t0_n)== v0_n ;
% const_p{end+1}= sp.getAccelT(t0_n)== a0_n ;
% 
% %Final conditions
% % opti.subject_to( sp.getPosT(tf)== pf );
% const_p{end+1}= sp.getVelT(tf_n)== vf_n ;
% const_p{end+1}= sp.getAccelT(tf_n)== af_n ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eq_const_init={};
eq_const_init{end+1}=sp.getPosT(t0_n)==p0;
eq_const_init{end+1}=sp.getVelT(t0_n)==v0_n;
eq_const_init{end+1}=sp.getAccelT(t0_n)==a0_n;

eq_cons_end={};
eq_cons_end{end+1}=sp.getVelT(tf_n)== vf_n;
eq_cons_end{end+1}=sp.getAccelT(tf_n)== af_n;


% sym_var_constraints=symvar(eq_constraints);
% intersection=intersect(variables,symvar(eq_constraints));
% solve(eq_constraints,variables(:,2:ncpts))
% solve(eq_constraints_init,variables(:,3) )

%See discussion here: https://www.mathworks.com/matlabcentral/answers/448087-is-there-a-way-to-specify-independent-variables-when-solving-symbolic-equations
solution_q1=solve(eq_const_init{1}, variables(:,1));
eq_const_init{2}=subs(eq_const_init{2},solution_q1);
solution_q2=solve(eq_const_init{2}, variables(:,2));
eq_const_init{3}=subs(eq_const_init{3},catstruct(solution_q1,solution_q2));
solution_q3=solve(eq_const_init{3}, variables(:,3));


solution_qN=solve(eq_cons_end{2}, variables(:,end));
eq_cons_end{1}=subs(eq_cons_end{1},solution_qN);
solution_qNm1=solve(eq_cons_end{1}, variables(:,end-1));

all_solutions=catstruct(solution_q1, solution_q2, solution_q3, solution_qN, solution_qNm1);

%%%%%%%%%%%%%%%%%%%%%55

dyn_lim_const=[];
dyn_lim_const=addDynLimConstraints(dyn_lim_const, sp, basis, v_max_n, a_max_n, j_max_n);


g=[];
for i=1:numel(dyn_lim_const)
   constraint=dyn_lim_const{i};
   constraint=subs(constraint, all_solutions);
   g=[g;lhs(constraint)-rhs(constraint)]; %Note that matlab uses lhs<=rhs (and never lhs>=rhs), see the answer in https://www.mathworks.com/matlabcentral/answers/267545-checking-whether-a-symbolic-expression-is-an-equality-or-inequality#comment_341853 
end

%Now I have g<=0

b=-subs(g, variables, zeros(size(variables)));
variables_flattened=variables(:);
A=jacobian(g,variables_flattened);
A=double(A);
%Now I have A*variables_flattened<=b
spy(A)

vpa(A*variables_flattened,2)

rows_A_are_zero=~any(A,2);  %rows
cols_A_are_zero=~any(A,1);  %rows

%Note that the constraints A(rows_A_are_zero,:)*variables<=b(rows_A_are_zero)  are guaranteed to be satisfied as long as |a0|<=a_max 

A(rows_A_are_zero,:)=[];
b(rows_A_are_zero)=[];
figure; spy(A)

A(:,cols_A_are_zero)=[];
variables_flattened(cols_A_are_zero)=[];
figure; spy(A)

Avmb=A*variables_flattened-b;

variables_flattened_value=rand(size(variables_flattened));
b_value=subs(b,[v0, a0], rand(3,2));
Avmb_value=A*variables_flattened_value-b_value;
f=matlabFunction(Avmb_value); %this needs to be <=0 (element-wise)

figure;
alpha_tmp=linspace(0,30,100);
plot(alpha_tmp, f(alpha_tmp),'k');
yline(0.0,'LineWidth',2,'Color','r');

%%

function result=getSym3DVector(name)
    x=sym(name+"_x",'real'); 
    y=sym(name+"_y",'real'); 
    z=sym(name+"_z",'real'); 
    result=[x;y;z];
end

function [const_p]=addDynLimConstraints(const_p, sp, basis, v_max_n, a_max_n, j_max_n)

   const_p=[const_p sp.getMaxVelConstraints(basis, v_max_n)];      %Max vel constraints (position)
   const_p=[const_p sp.getMaxAccelConstraints(basis, a_max_n)];    %Max accel constraints (position)
   const_p=[const_p sp.getMaxJerkConstraints(basis, j_max_n)];     %Max jerk constraints (position)

end

function result=containsSymCasadi(expression)
    result=(numel(symvar(expression))>0);
end

function result=isPlusInfCasadi(expression)
    if(containsSymCasadi(expression)) %if it has symbolic variables
        result=false;
        return;
    end
    result=(convertMX2Matlab(expression)==inf);
end

function result=isMinusInfCasadi(expression)
    if(containsSymCasadi(expression)) %if it has symbolic variables
        result=false;
        return;
    end
    result=(convertMX2Matlab(expression)==-inf);
end


% sp.updateCPsWithSolution(variables_SX)
% tmp=sp.getCPs_MV_Pos_ofInterval(1)
% v=tmp{1}; %First velocity control point
% % J=jacobian(v,variables);
% % J=convertMX2Matlab(J)
% 
% 
% Jx=convertMX2Matlab(jacobian(v(1),variables_SX))
% 
% %%
% vBS=sym('v%d_%d',[3,3],'real')'; %3 control points
% 
% vMV=vBS*getA_BS(2,[0,1])*inv(getA_MV(2,[0,1]));
% vpa(vMV,2)
% 
% %%
% 
% A=[];
% b=[];
% for i=1:size(A_tmp,1)
%     A=[A;A_tmp(i,:);-A_tmp(i,:)];
%     b=[b;opti_ubg(i);-lower(i)];
% end


%variables.reshape(1,:)
%variables=[sp.CPoints{3}]

% variables= SX.sym('x',size(variables,1),size(variables,2)); 
%variables=[variables(1,:)';variables(2,:)';variables(3,:)']


% A_tmp=[A_tmp;-A_tmp];
% b_tmp=[opti_ubg;-opti_lbg];
%Now I have A_tmp*variables<=b_tmp

%Convert to matlab stuff
% b=convertMX2Matlab(b);
% % A=convertMX2Matlab(A);
% 
% %This section below stores . TODO: more concise using matlab indexing A=A_tmp(b_tmp~=Inf,:)  ???? (casadi seems to complain about it)
% %A=[];b=[]; 
% % b_tmp=convertMX2Matlab(b_tmp);
% % for i=1:size(b_tmp,1)
% %     if b_tmp(i,1)~=Inf
% %         A=[A; A_tmp(i,:)];
% %         b=[b; b_tmp(i,:)];
% %     end
% % end
% 
% A=A(b~=Inf,:);
% b=b(b~=Inf,:);
% %%

% variables_SX=casadi.SX.sym('v',size(variables,1), size(variables,2));
% 
% compute_opti_g = Function('compute_opti_g', {variables} ,{opti_g}, {'var'} ,{'opti_g'});
% compute_opti_g=compute_opti_g.expand(); %It uses now SX
% 
% opti_g_SX=compute_opti_g(variables_SX);
% A_tmp=jacobian(opti_g_SX, variables_SX(:)); 
% 
% variables_xyz=[variables_SX(1,:), variables_SX(2,:), variables_SX(3,:)];
% 
% % A_tmp=convertMX2Matlab(A_tmp)
% 
% tmp_x=convertMX2Matlab(gradient(opti_g_SX(1),variables_xyz))
% tmp_y=convertMX2Matlab(gradient(opti_g_SX(1),variables_SX(2,:)))
% tmp_z=convertMX2Matlab(gradient(opti_g_SX(1),variables_SX(3,:)))
% 
% A_tmp=[];
% for i=1:1%size(opti_g,1)
%     all_row=[];
%     for j=1:size(variables,2)
%         
%         cp=variables_SX(:,j);
% %        term=subs(opti_g(i), )
%         tmp=[];
%         for u=1:size(cp,1)
%             tmp=gradient(opti_g_SX(i),cp(u));
%             all_row=[all_row, convertMX2Matlab(tmp)];
% %             tmp=convertMX2Matlab(tmp);
%         end
% %         all_row=[all_row tmp'];
%     end
%     A_tmp=[A_tmp; all_row];
% end

%%




% opti.subject_to(const_p);

%Note that the constraints are given by    opti.lbg() <= opti.g() <= opti.ubg()
% size(opti.g())
% 
% variables=[sp.getCPsAsMatrix()];
% 
% variables=casadi.SX.sym('v',size(variables,1), size(variables,2));
% 
% g=opti.g();
% lower=opti.lbg();
% upper=opti.ubg();
% 
% %Now I have    lower<=g<=upper
% 
% all_g=[];
% all_uper=[];
% 
% for i=1:size(g,1)
% 
%     if(isPlusInfCasadi(upper(i))==false)
%         all_uper=[all_uper; upper(i)];
%         all_g=[all_g; g(i)];
%     end
%     if(isMinusInfCasadi(lower(i))==false)
%         all_uper=[all_uper; -lower(i)];
%         all_g=[all_g; -g(i)];
%     end
% end
% 
% 
% 
% %Now I have all_g<=all_uper
% 
% %Obtain A and b
% % A_tmp=jacobian(opti_g, variables); %     opti.lbg() <= A_tmp*variables <= opti.ubg()
% % tmp=-casadi.substitute(opti_g, variables, zeros(size(variables))); %This should be zeros (i.e., opti_g should not contain scalar terms (without variables))
% 
% 
% opti_g_SX=convertExpresionWithMX2SX(all_g, variables, variables)
% 
% var_x=variables(1,:); var_y=variables(2,:); var_z=variables(3,:);
% var_xyz=[var_x, var_y, var_z];
% 
% % all_uper=convertMX2Matlab(all_uper);
% 
% Ax=[]; Ay=[]; Az=[];
% bx=[]; by=[]; bz=[];
% novale=[];
% for i=1:6:size(all_g,1)
%     i
%     all_row1=(gradient(opti_g_SX(i),var_xyz));
%     all_row2=(gradient(opti_g_SX(i+1),var_xyz));
% 
%     all_row3=(gradient(opti_g_SX(i+2),var_xyz));
%     all_row4=(gradient(opti_g_SX(i+3),var_xyz));
% 
%     all_row5=(gradient(opti_g_SX(i+4),var_xyz));
%     all_row6=(gradient(opti_g_SX(i+5),var_xyz));
%     
%     Ax=[Ax; all_row1; all_row2]; bx=[bx; all_uper(1:2,1)];
%     Ay=[Ay; all_row3; all_row4]; by=[by; all_uper(3:4,1)]; 
%     Az=[Az; all_row5; all_row6]; bz=[bz; all_uper(5:6,1)]; 
%     novale=[novale;all_row1;all_row2;all_row3;all_row4;all_row5;all_row6]
% end
% 
% 
% %Substitute here the parameters
% 
% v_max=opti.parameter(3,1);
% a_max=opti.parameter(3,1);
% j_max=opti.parameter(3,1);
% 
% alpha=opti.parameter(1,1); 
% 
% v_max_value=1.6*DM.ones(3,1);
% a_max_value=5*DM.ones(3,1);
% j_max_value=50*DM.ones(3,1);
% 
% 
% 
% A=[Ax; Ay; Az]; b=[bx; by; bz];
% 
% A=substitute(A,[v_max, a_max, j_max],[v_max_value, a_max_value, j_max_value])
% 
% spy(A)
% 
% %Now I have  A*var_xyz<=b
% 
% tmp=~all(Ax==0);
% Ax_nz=Ax(:,tmp) %Retain the columns that are nonzero
% b_nz=b(tmp)
% 
% save Ab.mat Ax_nz b_nz

% [V,nr,nre]=lcon2vert(Ax_nz,b_nz,[],[],[],false);%,TOL,True
% save('Ab.mat', {A,b})
%%

% A=A(:,4:end);
% 
% [V,nr,nre]=lcon2vert(A,b,[],[],[],false);%,TOL,True
% 
% %Now I have A*variables<=b
% 
% A =[0.4082   -0.8165    0.4082;
%     0.4082    0.4082   -0.8165;
%     -0.8165    0.4082    0.4082];
% b=[0.4082;
%     0.4082;
%     0.4082];
% 
% Aeq=[0.5774    0.5774    0.5774];
% beq=[0.5774];

%[V,nr,nre]=lcon2vert(A,b,Aeq,beq,[],true)%,TOL,True



