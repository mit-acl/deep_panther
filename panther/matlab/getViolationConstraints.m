function violation=getViolationConstraints(opti_tmp)

    g=opti_tmp.g();
    lower=opti_tmp.lbg();
    upper=opti_tmp.ubg();
    
    all_g=[];
    all_upper=[];
    
    for i=1:size(g,1)
        if(isPlusInfCasadi(upper(i))==false)
    %         upper(i)
            all_upper=[all_upper; upper(i)];
            all_g=[all_g; g(i)];
        end
        if(isMinusInfCasadi(lower(i))==false)
            all_upper=[all_upper; -lower(i)];
            all_g=[all_g; -g(i)];
        end
    end
    
    % The constraints are now all_g<=all_upper
    
    slack_constraints=all_g-all_upper; %If it's <=0 --> constraint is satisfied
    
    violation=[];%max()
    for i=1:size(slack_constraints,1)
        violation=[violation; max(0, slack_constraints(i))];
    end

end