syms x
d1=(cos(x)+sin(x))/sqrt(2);
d2=(cos(x)-sin(x))/sqrt(2);


term1=d1*cos(x)-d2*sin(x);
term2=d2*cos(x)+d1*sin(x);

simplify(term1-term2) %Should be 0
simplify(d1^2+d2^2) %Should be 1

simplify(term1)
% 
% simplify(acos(d1),'All',true)

simplify(d1-cos(x-pi/4))

simplify(d2-sin(x+3*pi/4))

d1=cos(x-pi/4);
d2=sin(x+3*pi/4);


Ra=[d1 -d2 0;
 d2  d1  0; 
 0   0   1]

Rb=[cos(x) -sin(x) 0;
    sin(x) cos(x) 0;
     0      0      1];
 
 Ra*Rb

% simplify(d2,'All',true)