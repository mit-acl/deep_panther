
%this function assumes that z_camera = b1
function b1=optimalb1FromPosPosFeatureAndAccel(w_p, w_feature, w_accel)

e=w_feature-w_p;
xi=w_accel+[0;0;9.81];

nxi2=xi'*xi; %norm of xi squared
nxi=sqrt(nxi2);


b1=(e-(e'*xi)*xi/nxi2); 

b1=b1/norm(b1);


end