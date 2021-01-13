function [C, out]=proj_delta(X,C)
%--------------------------------------------------------------------------
% projection onto the simplex defined by
% Delta = {C: diag(dot(X,C))=diag(I), C>=0}
%
% Input: X --- the matrix 
%        C --- Initial guess

out.success = 1; [m, n] = size(X); iter = 0;

% main loop
for i=1:n
    c = C(:,i);
    x = X(:,i);
    val_now = max(c,0)'*x;
    beta = 0.5;
    
    % estimate an approximate interval 
    if(val_now<1)
        lower_b = 0;
        c_l = c;
        while(val_now<=1)
            c = c+x*beta;
            val_now = max(c,0)'*x;
            beta = beta*2;
            iter = iter+1;
            % report error if input is ill-conditioned 
            if(beta>1e10)
                out.success = 0;
                return;
            end
        end
        upper_b = beta-0.5;
        c_u = c;
    else
        upper_b = 0;
        c_u = c;
        while(val_now>=1)
            c = c-x*beta;
            val_now = max(c,0)'*x;
            beta = beta*2;
            % report error if input is ill-conditioned 
            if(beta>1e10)
                out.success = 0;
                return;
            end
            iter =iter+1;
        end
        lower_b = -beta+0.5;
        c_l=c;
    end
    
    % search for the correct beta by divide-and-conquer method
    activate_u = sum(c_u>0);
    activate_l = sum(c_l>0);
    while(activate_l<activate_u)
        mid = (upper_b+lower_b)/2;
        c_m = C(:,i)+mid*x;
        val_now = max(c_m,0)'*x;
        if(val_now<1)
            c_l = c_m;
            lower_b = mid;
            activate_l = sum(c_l>0);
        else
            c_u = c_m;
            upper_b =mid;
            activate_u = sum(c_u>0);
        end
        iter =iter+1;
    end
    val_now = max(c_l,0)'*x;
    activate = c_l>0;
    C(:,i) = max(c_l+((1-val_now)/((x.*activate)'*x))*x,0);
    
end % end loop

% output
if(out.success==0); warning('Ill-conditioned projection problem'); end
out.iter = iter;

%-------------------------------------------------------------------------

end