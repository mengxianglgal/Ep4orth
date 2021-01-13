function [d,out]=cg_ssm(Jacobi_F,F,tau_cg,maxiter)
%--------------------------------------------------------------------------
% conjugate gradient method for solving d such that Jacobi_F(d)= F
%
% Input: 
% Jacobi_F,F --- The linear equation Jacobi_F(d)= F 
%     tau_cg --- Stopping criteria 
%    maxiter --- max number of iterations
% Output: 
%          d --- Solution of the linear equation
%        out --- Output information
%--------------------------------------------------------------------------
% initial setup

[n,k] = size(F);
d = zeros(n,k);
r = F - Jacobi_F(d);
rho = norm(r,'fro');
rho_0 = rho;
iter = 0;

% main loop
while(true)
    
    % check stopping criteria
    if(rho<tau_cg*(min(1,rho_0))); break; end
    iter = iter+1;
    if(iter>=maxiter); break; end

    if(iter==1)
        p = r;
    else
        beta = (rho^2)/(rho_^2);
        p = r+beta*p;
    end
    w = Jacobi_F(p);
    summ = sum(dot(p,w));
    
    % stop the cg method if we find d such that dot(d,Jacobi_F(d))<0
    if(summ<-1e-6*rho^2)
        out.neg = 1;
        out.success = 0;
        break;
    end
    
    alpha = rho^2/summ;
    d = d+alpha*p;
    r = r-alpha*w;
    rho_= rho;
    rho = norm(r,'fro');
end

%--------------------------------------------------------------------------
% store the iter. info.

out.iter = iter;
out.res = rho/rho_0;
out.neg = 0;
if(out.res>1)
    out.success = 0;
else
    out.success = 1;
end

%--------------------------------------------------------------------------

end