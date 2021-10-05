%%
% for j=1:sp.num_seg
%     
% 
%     w_t_b = sp.getPosU(u,j);
%     a=sp.getAccelU(u,j)/(alpha^(2));
%     xi=a+[0;0;g];
%     
%     %%%%
%       
%     if(use_yaw_closed_form==false)
%   
%         xi1=xi(1); xi2=xi(2); xi3=xi(3);
%         nxi2= xi1*xi1 + xi2*xi2 + xi3*xi3;
%         nxi=sqrt(nxi2);    
%         nxi2_plus_xi3nxi=nxi2 + xi(3)*nxi;
%         
%         yaw=sy.getPosU(u,j);
%         b1= [ (1-(xi1^2)/nxi2_plus_xi3nxi)*cos(yaw) -    xi1*xi2*sin(yaw)/nxi2_plus_xi3nxi;
%               -xi1*xi2*cos(yaw)/nxi2_plus_xi3nxi     +    (1 - (xi2^2)/nxi2_plus_xi3nxi)*sin(yaw);
%                (xi1/nxi)*cos(yaw)                 +    (xi2/nxi)*sin(yaw)         ]; %note that, by construction, norm(b1)=1;
%     
%     else
%     
%           b1=optimalb1FromPosPosFeatureAndAccel(w_t_b, zeros(3,1), a);
%     end
% 
%        
%     w_e=-w_t_b; %Asumming feature is in [0 0 0]';
%     
%     f=cos(thetax_half_FOV_deg*pi/180.0) - b1'*w_e/norm(w_e); %Constraint is f<=0
%     
%     fov_cost_j=max(0,f)^3; %Penalty associated with the constraint
%     
% 
% %     isInFOV=-cos(thetax_half_FOV_deg*pi/180.0) + b1'*w_e/norm(w_e);%This has to be >=0
% %     fov_cost_j=-isInFOV; Note that, if I use this, and isInFOV is
% %     -cos(XX)*norm(w_e) + b1'*w_e, then this part of the cost is unbounded
% %     (I can always keep decressing this term of the cost by modifying
% %     norm(w_e). As soon as other terms are added to the cost, this is
% %     fixed.
%     
%     %%%%%%%%%%%%%%%%%%
%       
%     span_interval=sp.timeSpanOfInterval(j);
%     
%     u_simpson{j}=getUSimpsonJ(span_interval, sp,j, t_simpson);    
%     
%     for u_i=u_simpson{j}
%                 
%         simpson_coeff=getSimpsonCoeff(simpson_index,num_samples_simpson);
%         fov_cost=fov_cost + (delta_simpson/3.0)*simpson_coeff*substitute( fov_cost_j,u,u_i); 
%         
%         all_fov_costs=[all_fov_costs substitute( fov_cost_j,u,u_i)];
%         
%         simpson_index=simpson_index+1;
%         
%     end
% end