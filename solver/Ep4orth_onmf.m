function [X, Y, out] = Ep4orth_onmf(A, X, opts, varargin)
%--------------------------------------------------------------------------
% An exact penalty problem for solving orthogonal nonnegative matrix
% factorization (ONMF)
%
%                  min     \|A-XY^T\|_F^2
%                  s.t.    X'X=I, X>=0, Y>=0.
%
% Input:
%         A --- The matrix to be decomposed
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
%               record: = 0, no print out
%
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

if nargin < 2
    error('at least two inputs: [X, Y, out] = Ep4orth_onmf(A,X)');
elseif nargin < 3
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
V = opts.V;

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
    fprintf('ONMF solver started... \n');
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
A = A/norm(A,'fro');
use_g = 1;

% main loop
for iter=1:maxiter
    
    % set parameters for subproblems in Ep4orth
    C = (X'*X)\X'*A;
    C = max(C,0);
    C_C = 2*(C*C');
    A_C =  2*(A*C');
    fval = @(x) -sum(dot(A_C,x))+sum(dot(C,(x'*x)*C));
    grad = @(x) (x*C_C-A_C);
    hess = @(x,d) d*C_C;
    subopts.tau = tau;
    subopts.tol = tol;
    
    % solve the penalty subproblem
    % switch to the second-order method if X is close to the feasible set
    if(norm(X*V,'fro')^2-1>altmethod && use_g==1)
        [X, subout] = penalf_ob(X,V,p,epsilon,sigma,...
            fval,grad,subopts);
    else
        [X, subout] = penals_ob(X,V,p,epsilon,sigma,...
            fval,grad,hess,subopts);
    end
    
    % store the iter. info. of inner iter.
    out.iter_sub(iter) = subout.iter; % inner iteration no.
    out.success(iter) = subout.success;
    out.feasi(iter) = subout.feasi-1;
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
    
    % if gradient step cannot decrease the function sufficiently, switch to
    % the second-order method
    if(subout.success==0); use_g = 0; end
    
    % ---- termination ----
    if(abs(subout.feasi-1)<=feasi_fin); break; end
    
end % end outer loop

% round solution X
if(abs(subout.feasi-1)<=rounding)
    X = round_st(X);
    out.post = 1;
    
    % postprocessing
    X2 = zeros(n,k);
    for i=1:k
        suppX = find(X(:,i)>0);
        [eigva,eigve] = svds(A(suppX,:)*A(suppX,:)',1);
        if(sum(eigva)<0); eigva = eigva*-1; end
        X2(suppX,i) = eigva;
    end
    
else
    out.post = 0;
end


%--------------------------------------------------------------------------
% store the iter. info.

if(hasRecordFile); fclose(fid); end
Y = A'*X;
out.iter = iter;
out.avgsubit = sum(out.iter_sub)/iter;
out.Fval = norm(A-X*Y','fro')^2;
out.time = toc(timetic);

%--------------------------------------------------------------------------

end