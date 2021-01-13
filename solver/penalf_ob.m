function [X, out] = penalf_ob(X,V,p,eps,sigma,fval,grad,opts,varargin)
%--------------------------------------------------------------------------
% Solve the penalty subproblem of the form
%
%                  min     f(X)+sigma*(||XV||_F^2-1+eps)^p
%                  s.t.    \|X_{:,i}\|_2=1, X>=0
%
% by projection gradient method with BB step size
%
% Input:
%             X --- Initial guess
% V p eps sigma --- Parameters of the penalty term
%     fval grad --- Objective f(X) and its gradient
%          opts --- Options structure with fields
%                   tol: stop control
%                   maxiter: max number of iterations
%                   ac_step: minimum acceptable stepsize (for line search)
%                   ac_ratio:  minimum acceptable rate of decline
%                   BB_type: type of BB stepsize (LBB/SBB/ABB)
%                   record: = 0, no print out
%
% Output:
%         X --- Solution
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

if nargin < 7
    error(['at least seven inputs: [X, out] = penalprob_ob(X,V,p'...
        ',eps sigma,fval,grad)']);
elseif nargin < 8
    opts = [];
end

%--------------------------------------------------------------------------
% options for solving the penalty subproblem

if ~isfield(opts,'tol');             opts.tol = 1e-2; end
if ~isfield(opts,'maxiter');         opts.maxiter = 20; end

if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
else; hasRecordFile = 0;
end
if ~isfield(opts,'record');          opts.record = 0; end
if ~isfield(opts,'itprint');         opts.itprint = 1; end
if ~isfield(opts,'printab');         opts.printab = 1; end

if ~isfield(opts,'ac_step');         opts.ac_step = 1e-2; end
if ~isfield(opts,'ac_ratio');        opts.ac_ratio = 1e-6; end
if ~isfield(opts,'init_s');          opts.init_s = 1; end
if ~isfield(opts,'gamma');           opts.gamma = 0.85; end
if ~isfield(opts,'BBtype');          opts.BBtype = 'SBB'; end
if ~isfield(opts,'maxBB');           opts.maxBB = inf; end

%--------------------------------------------------------------------------
% copy parameters

tol = opts.tol;         maxiter = opts.maxiter;   BBtype = opts.BBtype;
record = opts.record;   itprint = opts.itprint;   gamma = opts.gamma;
printab = opts.printab; ac_step = opts.ac_step;   ac_ratio = opts.ac_ratio;
init_s = opts.init_s;   maxBB = opts.maxBB;

%--------------------------------------------------------------------------
% prepare for recording iter. info.

stra = ['%7s','%13s','%13s','%11s','\n'];
str_head = sprintf(stra, ...
    'iter', 'f_val', 'feasi', ...
    'stepsize');
str_num = '   %4d  %+5.4e  %+5.4e  %+3.2e \n';

if(record)
    for i=1:printab; fprintf('\t'); end
    fprintf('Solving subproblem with constraint X in Ob_+(n,k) \n');
    for i=1:printab; fprintf('\t'); end
    fprintf('%s', str_head);
end

% record iter. info. as a file
if(hasRecordFile)
    for i=1:printab; fprintf(fid,'\t'); end
    fprintf(fid,'Solving subproblem with constraint X in Ob_+(n,k) \n');
    for i=1:printab; fprintf(fid, '\t'); end
    fprintf(fid, '%s', str_head);
end

%--------------------------------------------------------------------------
% initial setup

ptimetic = tic;
iter = 0; tot_sea = 0; flag = 1;
X_v = X*V; feasi = norm(X_v)^2; tmp_fea = feasi-1+eps;
fref = fval(X)+sigma*(tmp_fea^p); Q = 1;
pgrad = grad(X)+2*sigma*p*(tmp_fea^(p-1))*(X_v*V');

% main loop
while(true)
    iter = iter+1;
    
    % record previous iter.
    X_pre = X;
    G_pre = pgrad;
    step_BB = init_s;
    
    % non-monotone line search
    while(true)
        tot_sea = tot_sea+1;
        X = proj_ob(X_pre-step_BB*pgrad);
        X_v = X*V;
        feasi = norm(X*V)^2;
        tmp_fea = feasi-1+eps;
        fval_n = fval(X)+sigma*(tmp_fea^p);
        if(fval_n<fref-ac_ratio/step_BB*norm(X-X_pre,'fro')^2)
            break;
        else
            % if cannot find satisfying direction, switch to second-order
            % method
            if(step_BB<ac_step)
                out.success = 0;
                flag = 0;
                break;
            else
                step_BB = step_BB/2;
            end
        end
    end
    if(~flag); break; end
    % update parameters of non-monotone line search
    Qp = Q; Q = gamma*Qp+1; fref = (gamma*Qp*fref+fval_n)/Q;
    
    % compute gradient of the subproblem
    pgrad = grad(X)+2*sigma*p*(tmp_fea^(p-1))*(X_v*V');
    
    % calculate BB stepsize
    S = X-X_pre;
    N = pgrad-G_pre;
    switch BBtype
        case 'SBB'
            step_BB = min(maxBB,sum(dot(S,N))/norm(N,'fro')^2);
        case 'LBB'
            step_BB = min(maxBB,norm(S,'fro')^2/sum(dot(S,N)));
        case 'ABB'
            sn = sum(dot(S,N));
            if(mod(iter,2) == 0)
                step_BB = min(maxBB,sn/norm(N,'fro')^2);
            else
                step_BB = min(maxBB,norm(S,'fro')^2/sn);
            end
    end
    
    % ---- record ----
    if(record&&mod(iter,itprint)==0)
        for i=1:printab; fprintf('\t'); end
        fprintf(str_num,iter,fval_n,feasi-1,step_BB);
    end
    
    % record as a file
    if(hasRecordFile&&mod(iter,itprint)==0)
        for i=1:printab; fprintf(fid, '\t'); end
        fprintf(fid,str_num,iter,fval_n,feasi-1,step_BB);
    end
    
    % check stopping criteria
    if(iter>=maxiter); break; end
    if(norm(proj_ob(X-grad(X))-X,'fro')<tol&&iter>=3); break; end
    
end % end outer loop

%--------------------------------------------------------------------------
% store the iter. info.

if(hasRecordFile); fclose(fid); end
out.iter = iter;
out.search = tot_sea;
out.feasi = feasi;
out.success = 1;
out.time = toc(ptimetic);

%--------------------------------------------------------------------------

end
