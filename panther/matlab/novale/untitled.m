E=sym('E',[3 1])

q=(1/sqrt(2*(1+E(3))))*[1+E(3);-E(2);E(1);0]

assume(E(1)^2+E(2)^2+E(3)^2==1)
R=simplify(toRotMat(q))


R_paper=[1-(E(1)^2)/(1+E(3))     -E(1)*E(2)/(1+E(3))       E(1);
         -E(1)*E(2)/(1+E(3))     1-(E(2)^2)/(1+E(3))       E(2);
         -E(1)  -E(2)   E(3)];
     
simplify(R-R_paper)

function R= toRotMat(q)

q=q(:);

q0=q(1);
q1=q(2);
q2=q(3);
q3=q(4);

R=[q0^2+q1^2-q2^2-q3^2  2*q1*q2-2*q0*q3  2*q1*q3+2*q0*q2;
    2*q1*q2+2*q0*q3     q0^2-q1^2+q2^2-q3^2  2*q2*q3-2*q0*q1;
    2*q1*q3-2*q0*q2   2*q2*q3+2*q0*q1  q0^2-q1^2-q2^2+q3^2  
    ];


end