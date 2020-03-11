
%read 
%https://math.stackexchange.com/questions/1392916/smaller-enclosing-shape-for-b%C3%A9zier-curves
%Me acabo de encontrar este paper :(
%Polynomial bases for quadratic and cubic polynomials which yield control points with small convex hulls     

clear; clc; close all;
%set(0,'DefaultFigureWindowStyle','docked')
set(0,'DefaultFigureWindowStyle','normal')
%Useful to plot the result: http://nurbscalculator.in/

%READ THIS: https://yalmip.github.io/example/nonconvexquadraticprogramming/

W=[];
V=[];
for i=1:4
   W=[W sdpvar(2,2)];
   V=[V sdpvar(2,2)];
end

A = sdpvar(4,4,'full'); %Should I enforce A symmetric?? (for Bezier curves, it's symmetric)

constraints=[];
constraints=[constraints A*ones(4,1)==[0 0 0 1]'];%Sum \lambda_i(t)=1

U=[];
sum_Wi=zeros(2,2);
sum_Vi=zeros(2,2);
for i=1:4
    Wi=W(:,(2*i-1):2*i);
    Vi=V(:,(2*i-1):2*i);
    sum_Wi=sum_Wi+Wi;
    sum_Vi=sum_Vi+Vi;
    
    %Wi and Vi are psd matrices <=> All ppal minors are >=0
    %constraints=[constraints Wi>=0 Vi>=0];
    constraints=[constraints, (Wi(1,1)>=0):'Wi11>=0', (Vi(1,1)>=0):'Vi11>=0'];
    constraints=[constraints, ((Wi(1,1)*Wi(2,2)-Wi(1,2)*Wi(1,2))>=0):'(Wi(1,1)*Wi(2,2)-Wi(1,2)*Wi(1,2)>=0'];
    constraints=[constraints, ((Vi(1,1)*Vi(2,2)-Vi(1,2)*Vi(1,2))>=0):'Vi(1,1)*Vi(2,2)-Vi(1,2)*Vi(1,2)>=0'];
    constraints=[constraints, (Wi(2,2)>=0):'Wi22>=0', (Vi(2,2)>=0):'Vi22>=0'];
    %%%%%%%
    
    ui=[Wi(2,2)-Vi(2,2)   ,  -2*Vi(1,2)+Vi(2,2)+2*Wi(1,2) ,  -Vi(1,1)+2*Vi(1,2)+Wi(1,1) , Vi(1,1)]';
    U=[U; ui'];
end

constraints=[constraints A'==U]; %
C=sum_Wi-sum_Vi;
D=sum_Vi;

constraints=[constraints, C(2,2)==0];
constraints=[constraints, C(1,2)+C(2,1)+D(2,2)==0];
constraints=[constraints, C(1,1)+D(1,2)+D(2,1)==0];
constraints=[constraints, D(1,1)==1];

% constraints=[constraints, A(2,4)==0, A(3,4)==0, A(4,4)==0 , A(3,3)==0 , A(4,3)==0 ];

a11=A(1,1); a12=A(1,2); a13=A(1,3); a14=A(1,4);
a21=A(2,1); a22=A(2,2); a23=A(2,3); a24=A(2,4);
a31=A(3,1); a32=A(3,2); a33=A(3,3); a34=A(3,4);
a41=A(4,1); a42=A(4,2); a43=A(4,3); a44=A(4,4);

determinant=a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41;

%I want to maximize the absolute value of the determinant of A
obj=-(determinant)*determinant

%I want to maximize the absolute value of the determinant of A
%obj=-abs(det(A,'polynomial')); %Should I put abs() here?

W_bezier=[0.0051   -0.0051    3.0000   -3.0003    0.0005   -0.0005    0.0000   -0.0022;
         -0.0051    0.0051   -3.0003    3.0006   -0.0005    0.0005   -0.0022    1.0043];

V_bezier=[1.0000   -1.0025    0.0000   -0.0000    0.0000   -0.0003    0.0000   -0.0000
       -1.0025    1.0051   -0.0000    0.0006   -0.0003    3.0005   -0.0000    0.0043];
     
A_bezier=[-1 3 -3 1;
         3 -6 3 0;
         -3 3 0 0;
         1 0 0 0];
assign(A,A_bezier);
assign(U,A_bezier');
assign(W,W_bezier);
assign(V,V_bezier);



% disp('WWWWWWWWWWWWWWWWWWWWWW')
% constraints_novale=[]
% for i=1:4
%     Wi=W_bezier(:,(2*i-1):2*i);
%     Vi=W_bezier(:,(2*i-1):2*i);
%     
%     Wi(1,1)
%     Vi(1,1)
%     Wi(1,1)*Wi(2,2)-Wi(1,2)*Wi(1,2)
%     Vi(1,1)*Vi(2,2)-Vi(1,2)*Vi(1,2)
%     constraints_novale=[constraints_novale, (Wi(1,1)>=0), (Vi(1,1)>=0)];
%     constraints_novale=[constraints_novale, (Wi(1,1)*Wi(2,2)-Wi(1,2)*Wi(1,2)>=0)];
%     constraints_novale=[constraints_novale, (Vi(1,1)*Vi(2,2)-Vi(1,2)*Vi(1,2)>=0)];
% end
% disp('WWWWWWWWWWWWWWWWWWWWWW')


disp('Starting optimization') %'solver','bmibnb'  ,'solver','sdpt3' 'ipopt' 'knitro'
result=optimize(constraints,obj,sdpsettings('usex0',1,'solver','fmincon','showprogress',1,'verbose',2,'debug',0,'fmincon.maxfunevals',300000 ));
check(constraints)

A_value=value(A);
U_value=value(U);
W_value=value(W);
V_value=value(V);

sum_Vi_value=value(sum_Vi);
sum_Wi_value=value(sum_Wi);
C_value=value(C);
D_value=value(D);

W1=W_value(:,1:2); V1=V_value(:,1:2);
W2=W_value(:,3:4); V2=V_value(:,3:4);
W3=W_value(:,5:6); V3=V_value(:,5:6);
W4=W_value(:,7:8); V4=V_value(:,7:8);

t=0.8;

[1 t]*(t*C_value + D_value)*[1;t]

%% 
figure
syms t real
T=[t*t*t t*t t 1]';
t2=[1 t]';
lambda1= A_value(:,1)'*T;
lambda2= A_value(:,2)'*T;
lambda3= A_value(:,3)'*T;
lambda4= A_value(:,4)'*T;
fplot(lambda1,[0,1]); hold on;
fplot(lambda2,[0,1]);
fplot(lambda3,[0,1]);
fplot(lambda4,[0,1]);
xlim([0 1])

temporal=t2'*(t*W3 + (1-t)*V3)*t2; %should be lambda1
coeff_temporal=vpa(coeffs(temporal,t),4)

coeff_lambda3=vpa(coeffs(lambda3,t),4)

%%
pol_x=[0.2 0.3 2 1]';%[a b c d]
pol_y=[-0.3 +3 -5 6]';%[a b c d]
pol_z=[1 -0.1 -1 -4]';%[a b c d]


% px=5*t*t*t+7*t*t+2*t+1;
% py=-4*t*t*t+6*t*t+5*t+6;
% pz=10*t*t*t+2*t*t+3*t+4;

figure;
subplot(2,1,1);
fplot(pol_x'*T,pol_y'*T,[0 1],'r','LineWidth',3)
xlabel('x'); ylabel('y');
subplot(2,1,2);
fplot(pol_x'*T,pol_z'*T,[0 1],'r','LineWidth',3)
xlabel('x'); zlabel('z');

%%
figure; hold on;
fplot3(pol_x'*T,pol_y'*T,pol_z'*T,[0 1],'r','LineWidth',3);
%axis equal
volumen_mio=plot_convex_hull(pol_x,pol_y,pol_z,A_value,'b');
volumen_bezier=plot_convex_hull(pol_x,pol_y,pol_z,A_bezier,'g');
disp("abs(|A_mio|/|A_bezier|)=")
abs(det(A_value)/det(A_bezier))
disp("volumen_bezier/volumen_mio=")
volumen_bezier/volumen_mio

function volume=plot_convex_hull(pol_x,pol_y,pol_z,A,color)
    cx=pol_x;
    cy=pol_y;
    cz=pol_z;

    vx=inv(A)*cx;
    vy=inv(A)*cy;
    vz=inv(A)*cz;

    v1=[vx(1) vy(1) vz(1)]';
    v2=[vx(2) vy(2) vz(2)]';
    v3=[vx(3) vy(3) vz(3)]';
    v4=[vx(4) vy(4) vz(4)]';

    plot3(v1(1),v1(2),v1(3),'-o','Color',color,'MarkerSize',10)
    plot3(v2(1),v2(2),v2(3),'-o','Color',color,'MarkerSize',10)
    plot3(v3(1),v3(2),v3(3),'-o','Color',color,'MarkerSize',10)
    plot3(v4(1),v4(2),v4(3),'-o','Color',color,'MarkerSize',10)

    [k1,volume] = convhull(vx,vy,vz);

    trisurf(k1,vx,vy,vz,'FaceColor',color)
    xlabel('x')
    ylabel('y')
    zlabel('z')
    alpha 0.2
    %axis equal
end
