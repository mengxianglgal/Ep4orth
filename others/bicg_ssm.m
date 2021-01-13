function [d, out]=bicg_ssm(Jacobi_F,F,tau_cg,max_cgite)
%--------------------------------------------------------------------------
% biconjugate gradient method for solving d such that Jacobi_F(d)= F
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

[n, k] = size(F);
d = zeros(n,k);
r = F - Jacobi_F(d);
r_ = r;
rho_p = 1; alpha = 1; omega = 1;
p = zeros(n,k); v = zeros(n,k);
rho = norm(r,'fro');
rho_0 = rho;
tot_step = 0;

% main loop
while(true) 
    tot_step = tot_step+1;
    if(tot_step>=max_cgite); break; end

    rho = sum(dot(r_,r));
    beta = rho*alpha/(omega*rho_p);
    p = r + beta*(p-omega*v);
    v = Jacobi_F(p);
    alpha = rho/sum(dot(r_,v));
    h = d+alpha*p;
    res = norm(F-Jacobi_F(h),'fro');
    
    % check stopping criteria
    if(res<tau_cg*(min(1,rho_0))); d = h; break; end

    s = r-alpha*v;
    t = Jacobi_F(s);
    omega = sum(dot(s,t))/norm(t,'fro')^2;
    d = h+omega*s;
    res = norm(F-Jacobi_F(d),'fro');
    
    % check stopping criteria
    if(res<tau_cg*(min(1,rho_0))); break; end
    
    r = s-omega*t;
    rho_p = rho;
end

%--------------------------------------------------------------------------
% store the iter. info.

out.step = tot_step;
out.res = res/rho_0;
out.neg = 0;
if(out.res>1)
    out.success = 0;
else
    out.success = 1;
end

%--------------------------------------------------------------------------

end