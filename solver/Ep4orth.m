function [X, out] = Ep4orth(fval, grad, hess, X, opts, varargin)
%--------------------------------------------------------------------------
% An exact penalty problem for solving optimization problems with
% orthogonal and nonnegative constraints
%
%                  min     f(X)
%                  s.t.    X'X=I, X>=0
%
% Input:
%      fval --- The objective of f
%      grad --- grad(X) returns \nabla f(X)
%      hess --- hess(X,d) returns \nabla^2 f(X)[d]
%         X --- Initial guess
%      opts --- Options structure with fields
%               V p epsilon sigma: parameters for the penalty term
%                       sigma*(||XV||_F^2-1+epsilon)^p
%               tau: regularization term for the second-order method when
%                       solving the penalty subproblem
%               tol: stop control for the subproblem
%               feasi_fin: stop control for the constraint violation
%               eps_min sigma_max tol_min: maximum/minimum constraints
%                       for epsilon, simga and tol, respectively.
%               omega1 omega2 omega3: parameters for adjusting the
%                       penalty parameters
%               maxiter: max number of iterations
%               altmethod: time to switch to the second-order method
%               subopts: options for the penalty subproblem
%               objtype: = 0/1, control the type of penalty
%                       =0, f(X)+sigma*penal, =1, f(X)/sigma+penal
%               record: = 0, no print out
% Output:
%      X, Y --- Solutions
%       out --- Output information
%--------------------------------------------------------------------------
% Reference:
% B. Jiang, X. Meng, Z. Wen and X. Chen
% An Exact Penalty Approach For Optimization With Nonnegative Orthogonality
% Constraints
%
% Author: X. Meng, B. Jiang
% Version 1.0 .... 2021/1

%--------------------------------------------------------------------------

if nargin < 4
    error('at least four inputs: [X, out] = Ep4orth(fval,grad,hess,X)');
elseif nargin < 5
    opts = [];
end

% size of the problem
[n, k] = size(X);

%--------------------------------------------------------------------------
% options for the ONMF solver

if ~isfield(opts,'tau');             opts.tau = 5e-2; end
if ~isfield(opts,'p');               opts.p = 1; end
if ~isfield(opts,'V');               opts.V = ones(k,1)/sqrt(k); end

if ~isfield(opts,'epsilon');         opts.epsilon = 0; end
if ~isfield(opts,'eps_min');         opts.eps_min = 0; end
if ~isfield(opts,'sigma');           opts.sigma = 1e-3; end
if ~isfield(opts,'sigma_max');       opts.sigma_max = 1e5; end
if ~isfield(opts,'tol');             opts.tol = 1e-1; end
if ~isfield(opts,'tol_min');         opts.tol_min = 1e-7; end
if ~isfield(opts,'feasi_fin');       opts.feasi_fin = 1e-8; end

if ~isfield(opts,'omega1');          opts.omega1 = 0.98; end
if ~isfield(opts,'omega2');          opts.omega2 = 1.05; end
if ~isfield(opts,'omega3');          opts.omega3 = 0.98; end

if ~isfield(opts,'maxiter');         opts.maxiter = 3e2; end
if ~isfield(opts,'objtype');         opts.objtype = 0; end
if ~isfield(opts,'altmethod');       opts.altmethod = inf; end
if ~isfield(opts,'subopts');         opts.subopts = []; end

if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
else; hasRecordFile = 0; 
end
if ~isfield(opts, 'record');         opts.record = 0; end
if ~isfield(opts,'itprint');         opts.itprint = 1; end
if ~isfield(opts,'printab');         opts.printab = 0; end

%--------------------------------------------------------------------------
% copy parameters

tau = opts.tau;         p = opts.p;             epsilon = opts.epsilon;
eps_min = opts.eps_min; sigma = opts.sigma;     sigma_max = opts.sigma_max;
tol = opts.tol;         tol_min = opts.tol_min; feasi_fin = opts.feasi_fin;
record = opts.record;   itprint = opts.itprint; printab = opts.printab;
subopts = opts.subopts; maxiter = opts.maxiter; altmethod = opts.altmethod;
omega1 = opts.omega1;   omega2 = opts.omega2;   omega3 = opts.omega3;
V = opts.V;             objtype = opts.objtype;
 
%--------------------------------------------------------------------------
% prepare for recording iter. info.

stra = ['%4s','%15s','%15s','%10s','%11s','%9s','%10s','%10s','\n'];
str_head = sprintf(stra, ...
    'iter','f_value','feasi','eps', ...
    'sigma','tol','sub_iter','sub_sea');
str_num = ['%4d    %+5.4e    %+5.4e  %+2.1e ' ...
    '  %+2.1e %+2.1e     %5d  %+2.1e \n'];

if(record)
    for i=1:printab; fprintf('\t'); end
    fprintf('Ep4orth solver started... \n');
    if(~(isfield(subopts,'record')&&subopts.record==1))
        for i=1:printab; fprintf('\t'); end
        fprintf('%s', str_head);
    end
end

% record iter. info. as a file
if(hasRecordFile&&~isfield(subopts,'recordFile'))
    for i=1:printab; fprintf(fid, '\t'); end
    fprintf(fid, '%s', str_head);    
end


%--------------------------------------------------------------------------
% initial setup

timetic = tic;
rounding = min(min(V*V'));
use_g = 1; fea_fail = 0;
Xinit = X; feasi_pre = Inf;


% main loop
for iter=1:maxiter
    
    subopts.tau = tau;
    subopts.tol = tol;
    
    % solve the penalty subproblem
    % switch to the second-order method if X is close to the feasible set
    if(norm(X*V,'fro')^2-1>altmethod && use_g==1)
        if(objtype)
            fval2 = @(X) fval(X)/sigma;
            grad2 = @(X) grad(X)/sigma;
            [X, subout] = penalf_ob(X,V,p,epsilon,1,...
                fval2,grad2,subopts);
        else
            [X, subout] = penalf_ob(X,V,p,epsilon,sigma,...
                fval,grad,subopts);
        end
    else
        if(objtype)
            fval2 = @(X) fval(X)/sigma;
            grad2 = @(X) grad(X)/sigma;
            hess2 = @(X,d) hess(X,d)/sigma;
            [X, subout] = penals_ob(X,V,p,epsilon,1,...
                fval2,grad2,hess2,subopts);
        else
            [X, subout] = penals_ob(X,V,p,epsilon,sigma,...
                fval,grad,hess,subopts);
        end
    end
    
    % store the iter. info. of inner iter.
    out.iter_sub(iter) = subout.iter; % inner iteration 
    out.success(iter) = subout.success;
    out.feasi(iter) = subout.feasi;
    out.search(iter) = subout.search;
    out.time_sub(iter) = toc(timetic);
    if(isfield(subout,'tau')); tau = subout.tau; end
    
    % ---- record ----
    if(record&&mod(iter,itprint)==0)
        if(isfield(subopts,'record')&&subopts.record==1)
            for i=1:printab; fprintf('\t'); end
            fprintf('%s', str_head);
        end
        for i=1:printab; fprintf('\t'); end
        fprintf(str_num,iter,fval(X),subout.feasi-1,...
            epsilon,sigma,tol,subout.iter,subout.search/(subout.iter));
    end
    
    % record as a file
    if(hasRecordFile&&mod(iter,itprint)==0)
        if(isfield(subopts,'recordFile'))
            for i=1:printab; fprintf(fid, '\t'); end
            fprintf(fid, '%s', str_head);
        end
        for i=1:printab; fprintf(fid, '\t'); end
        fprintf(fid, str_num,iter,fval(X),subout.feasi-1,...
            epsilon,sigma,tol,subout.iter,subout.search/(subout.iter));    
    end
    
    % update parameters
    epsilon = max(eps_min,epsilon*omega1);
    sigma = min(sigma_max,sigma*omega2);
    tol = max(tol_min,tol*omega3);
    
    % if feasi do not decrease sufficiently, increase fea_fail
    if(subout.feasi>0.95*feasi_pre)
        fea_fail =  fea_fail+1;
    end
    feasi_pre = subout.feasi;
    
    % round X to a feasible point and set it to be the initial point 
    [X_tmp, can_proj] = round_st(X);
    if(can_proj);  Xinit = X_tmp; end

    % if the feasibility does not decrease (stuck at saddle), reset X
    if (fea_fail>=5)
        fea_fail = 0;
        X = Xinit;
    end
    
    % if gradient step cannot decrease the function sufficiently, switch to
    % the second-order method
    if(subout.success==0); use_g = 0; end
    
    % ---- termination ----
    if(abs(subout.feasi-1)<=feasi_fin); break; end
    
end % end outer loop

% round solution X
if(abs(subout.feasi-1)<=rounding); X = round_st(X); end

%--------------------------------------------------------------------------
% store the iter. info.

if(hasRecordFile); fclose(fid); end
out.iter = iter;
out.avgsubit = sum(out.iter_sub)/iter;
out.Fval = fval(X);
out.time = toc(timetic);

%--------------------------------------------------------------------------

end