clear; close all; clc;

syms t real;
T=[t^3; t^2; t;1];
p = sym('p', [4 1],'real');
pt=p'*T;
at =diff(pt,2,t);

f=at*pt

%I want to write f as f= p'Gp + b'p + c. Note that b and c will be zero in this case

G=simplify(hessian(f,p))/2.0;

simplify(f-p'*G*p)


%%

for ti=3
   
   Gi=subs(G,t,ti);
   double( eig(Gi))
    
end

for ti=3
   
   fi=subs(f,t,ti);
   hessian_i=hessian(fi,p);
   double( eig(hessian_i))
    
end

% 
% 
% A=[0 0 0 0;
%    0 0 0 0;
%    6 0 0 0;
%    0 2 0 0];
% 
% A
% 
% p'*A'
% 
% at
% 
% A'*T*T'
% 
% simplify(f-p'*A'*T*T'*p)