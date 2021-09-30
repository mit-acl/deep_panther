% /* ----------------------------------------------------------------------------
%  * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
%  * Massachusetts Institute of Technology
%  * All Rights Reserved
%  * Authors: Jesus Tordesillas, et al.
%  * See LICENSE file for the license information
%  * -------------------------------------------------------------------------- */

%Everything here is 1-based indexing (first element is one)

%See https://www.mathworks.com/help/matlab/matlab_oop/comparing-handle-and-value-classes.html
classdef MyCasadiClampedUniformSpline < handle

    properties
        t0
        tf
        delta_t
        M
        N
        p   
        knots
        num_seg
        num_cpoints
        CPoints %Matrix of size dim\times(N+1)
        dim
        expression
        t
        getPosT
        
    end
    
    methods
        function obj = MyCasadiClampedUniformSpline(t0, tf, deg, dim, num_seg, CPoints, short_circuit) 
            
            %For short_circuit, see https://groups.google.com/g/casadi-users/c/KobfQ47ZAG8
            obj.dim=dim;
            obj.t0 = t0;
            obj.tf = tf;
            obj.p = deg;
            obj.num_seg = num_seg;
            obj.M =  obj.num_seg + 2 * obj.p;
            obj.delta_t = (obj.tf -  obj.t0) / (1.0 * (obj.M - 2 * obj.p - 1 + 1));
            obj.N = obj.M - obj.p - 1;
            obj.num_cpoints=obj.N+1;

            obj.knots=[obj.t0*ones(1,obj.p+1)       obj.t0+obj.delta_t*(1:obj.M - 2*obj.p-1)          obj.tf*ones(1,obj.p+1)];

            obj.expression=[];
            obj.CPoints=CPoints; %Matrix of size dim\times(N+1)
            obj.t=casadi.MX.sym('t',1,1);
            
            for j=(obj.num_seg):-1:1 %j is the index of the segment

                t_end=j*obj.delta_t;
                t_init=(j-1)*obj.delta_t;

                interval=[0,1];
                A=computeMatrixForAnyBSpline(deg,deg+j,obj.knots,interval);
                V=obj.CPoints(:, j:(j+deg));

                

                if(j<obj.num_seg)
                    
                    u=(obj.t-t_init)/(t_end-t_init);
                    Pt=V*A*getT(deg,u);
                    
                    obj.expression=if_else( (t_init<=obj.t & obj.t<t_end), Pt, obj.expression, short_circuit);         
                else
                    
%                     u=min((obj.t-t_init)/(t_end-t_init),1.0);
                    
                    u=(obj.t-t_init)/(t_end-t_init);
                    Pt=V*A*getT(deg,u);
                    
                    obj.expression=Pt; %This is the last segment
                end

            end
            
            obj.getPosT=casadi.Function('f',{obj.t},{obj.expression});

        end

 
    end
end
