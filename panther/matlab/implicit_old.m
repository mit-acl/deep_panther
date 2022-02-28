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
tf_n=2.0;
deg_pos=3;
dim_pos=3;
num_seg =6; %number of segments

sp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.

v_max_n=1.0*ones(3,1);
a_max_n=1.0*ones(3,1);
j_max_n=1.0*ones(3,1);

const_p=[];
const_p=addDynLimConstraints(const_p, sp, basis, v_max_n, a_max_n, j_max_n);

opti.subject_to(const_p);

%Note that the constraints are given by    opti.lbg() <= opti.g() <= opti.ubg()
size(opti.g())

variables=[sp.getCPsAsMatrix()];

variables_SX=casadi.SX.sym('v',size(variables,1), size(variables,2));

g=opti.g();
lower=opti.lbg();
upper=opti.ubg();

%Now I have    lower<=g<=upper

all_g=[];
all_uper=[];

for i=1:size(g,1)
    if(convertMX2Matlab(upper(i))~=inf)
        all_uper=[all_uper; upper(i)];
        all_g=[all_g; g(i)];
    end
    if(convertMX2Matlab(lower(i))~=-inf)
        all_uper=[all_uper; -lower(i)];
        all_g=[all_g; -g(i)];
    end
end

%Now I have all_g<=all_uper

%Obtain A and b
% A_tmp=jacobian(opti_g, variables); %     opti.lbg() <= A_tmp*variables <= opti.ubg()
% tmp=-casadi.substitute(opti_g, variables, zeros(size(variables))); %This should be zeros (i.e., opti_g should not contain scalar terms (without variables))


opti_g_SX=convertExpresionWithMX2SX(all_g, variables, variables_SX)

var_x=variables_SX(1,:); var_y=variables_SX(2,:); var_z=variables_SX(3,:);
var_xyz=[var_x, var_y, var_z];

all_uper=convertMX2Matlab(all_uper);

Ax=[]; Ay=[]; Az=[];
bx=[]; by=[]; bz=[];
novale=[];
for i=1:6:size(all_g,1)
    i
    all_row1=convertMX2Matlab(gradient(opti_g_SX(i),var_xyz));
    all_row2=convertMX2Matlab(gradient(opti_g_SX(i+1),var_xyz));

    all_row3=convertMX2Matlab(gradient(opti_g_SX(i+2),var_xyz));
    all_row4=convertMX2Matlab(gradient(opti_g_SX(i+3),var_xyz));

    all_row5=convertMX2Matlab(gradient(opti_g_SX(i+4),var_xyz));
    all_row6=convertMX2Matlab(gradient(opti_g_SX(i+5),var_xyz));
    
    Ax=[Ax; all_row1; all_row2]; bx=[bx; all_uper(1:2,1)];
    Ay=[Ay; all_row3; all_row4]; by=[by; all_uper(3:4,1)]; 
    Az=[Az; all_row5; all_row6]; bz=[bz; all_uper(5:6,1)]; 
    novale=[novale;all_row1;all_row2;all_row3;all_row4;all_row5;all_row6]
end


A=[Ax; Ay; Az]; b=[bx; by; bz];
spy(A)

%Now I have  A*var_xyz<=b

tmp=~all(Ax==0);
Ax_nz=Ax(:,tmp) %Retain the columns that are nonzero
b_nz=b(tmp)

save Ab.mat Ax_nz b_nz

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

function [const_p]=addDynLimConstraints(const_p, sp, basis, v_max_n, a_max_n, j_max_n)

   const_p=[const_p sp.getMaxVelConstraints(basis, v_max_n)];      %Max vel constraints (position)
   const_p=[const_p sp.getMaxAccelConstraints(basis, a_max_n)];    %Max accel constraints (position)
   const_p=[const_p sp.getMaxJerkConstraints(basis, j_max_n)];     %Max jerk constraints (position)

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



