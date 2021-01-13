function [X, out] = penals_ob(X,V,p,eps,sigma,fval,grad,hess,opts,varargin)
%--------------------------------------------------------------------------
% Solve the penalty subproblem of the form
%
%                  min     f(X)+sigma*(||XV||_F^2-1+eps)^p
%                  s.t.    \|X_{:,i}\|_2=1, X>=0
%
% by second-order method + line search
%
% Input:
%              X --- Initial guess
%  V p eps sigma --- Parameters of the penalty term
% fval grad hess --- Objective f(X), its gradient and hessian
%           opts --- Options structure with fields
%                    tau: regularization term for the second-order method
%                           when solving the penalty subproblem
%                    tau_min tau_max: value range of tau
%                    eta1 eta2 gamm1 gamma2 gamma3: para. for updating tau
%                    alpha: parameter term for the second-order method
%                           when solving the penalty subproblem
%                    tol: stop control
%                    maxiter: max number of iterations
%                    submaxit: max number of iterations for the
%                           second-order method
%                    m_a m_b: parameters for the line search
%                    ac_ratio:  minimum acceptable rate of decline
%                    record: = 0, no print out
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
% J. Hu, A. Milzarek, Z. Wen, and Y. Yuan, 
% Adaptive quadratically regularized Newton method for Riemannian 
% Optimization
%
%
% Author: X. Meng, B. Jiang
% Version 1.0 .... 2021/1

%--------------------------------------------------------------------------

if nargin < 8
    error(['at least eight inputs: [X, out] = penalprob_ob(X,V,p'...
        ',eps sigma,fval,grad,hess)']);
elseif nargin < 9
    opts = [];
end

%--------------------------------------------------------------------------
% options for solving the penalty subproblem

if ~isfield(opts,'tau');             opts.tau = 5e-2; end
if ~isfield(opts,'alpha');           opts.alpha = 1; end
if ~isfield(opts,'tau_min');         opts.tau_min = 5e-8; end
if ~isfield(opts,'tau_max');         opts.tau_max = 1e3; end

if ~isfield(opts,'eta1');            opts.eta1 = 1e-2; end
if ~isfield(opts,'eta2');            opts.eta2 = 0.9; end
if ~isfield(opts,'gamma1');          opts.gamma1 = 0.98; end
if ~isfield(opts,'gamma2');          opts.gamma2 = 1; end
if ~isfield(opts,'gamma3');          opts.gamma3 = 1.3; end

if ~isfield(opts,'tol');             opts.tol = 1e-2; end
if ~isfield(opts,'maxiter');         opts.maxiter = 20; end
if ~isfield(opts,'submaxit');        opts.submaxit = 20; end
if ~isfield(opts,'subopts');         opts.subopts = []; end

if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
else; hasRecordFile = 0;
end
if ~isfield(opts,'record');          opts.record = 0; end
if ~isfield(opts,'itprint');         opts.itprint = 1; end
if ~isfield(opts,'printab');         opts.printab = 1; end

if ~isfield(opts,'ac_step');         opts.ac_step = 1e-2; end
if ~isfield(opts,'init_s');          opts.init_s = 1; end
if ~isfield(opts,'m_a');             opts.m_a = 5e-3; end
if ~isfield(opts,'m_b');             opts.m_b = 1e3; end

%--------------------------------------------------------------------------
% copy parameters

tau = opts.tau;         tau_min = opts.tau_min; tau_max = opts.tau_max;
eta1 = opts.eta1;       eta2 = opts.eta2;       gamma1 = opts.gamma1;
gamma2 = opts.gamma2;   gamma3 = opts.gamma3;   tol = opts.tol;
subopts = opts.subopts; maxiter = opts.maxiter; record = opts.record;
itprint = opts.itprint; printab = opts.printab; ac_step = opts.ac_step;
init_s = opts.init_s;   m_a = opts.m_a;         m_b = opts.m_b;
alpha = opts.alpha;     submaxit = opts.submaxit;

%--------------------------------------------------------------------------
% prepare for recording iter. info.

stra = ['%7s','%13s','%13s','%11s','%11s','%12s','%8s','%11s','%8s','\n'];
str_head = sprintf(stra, ...
    'iter', 'f_val', 'feasi','tau', ...
    'rho','ratio','sub_it','tol','success');
str_num = ['   %4d  %+5.4e  %+5.4e  %+3.2e ' ...
    '  %+2.1e   %+3.2e    %4d  %+3.2e %7d\n'];

if(record)
    for i=1:printab; fprintf('\t'); end
    fprintf('Solving subproblem with constraint X in Ob_+(n,k) \n');
    if(isfield(subopts,'record')&&subopts.record==0)
        for i=1:printab; fprintf('\t'); end
        fprintf('%s', str_head);
    end
end

% record iter. info. as a file
if(hasRecordFile)
    for i=1:printab; fprintf(fid,'\t'); end
    fprintf(fid,'Solving subproblem with constraint X in Ob_+(n,k) \n');
    if(~isfield(subopts,'recordFile'))
        for i=1:printab; fprintf(fid, '\t'); end
        fprintf(fid, '%s', str_head);
    end
end

%--------------------------------------------------------------------------
% initial setup

ptimetic = tic;
iter = 0; tot_sea = 0; success = 1;
X_v = X*V; X_vv = X_v*V'; feasi = norm(X_v)^2;
tmp_fea = feasi-1+eps; pfval = fval(X)+sigma*(tmp_fea^p);

% main loop
while(true)
    iter = iter+1;
    
    % compute (Riemann) gradient and hessian of subproblem
    pgrad = grad(X)+2*sigma*p*(tmp_fea^(p-1))*(X_v*V');
    XGdot = dot(X,pgrad);
    pRgrad = pgrad-X.*XGdot;
    if(p~=1)
        pRhess = @(X,d) hess(X,d)+2*sigma*p*(tmp_fea^(p-1))*((d*V)*V')+...
            2*sigma*p*(p-1)*(tmp_fea^(p-2))*(V'*d'*X_v)*(X_vv)-d.*XGdot;
        phess = @(X,d) hess(X,d)+2*sigma*p*(tmp_fea^(p-1))*((d*V)*V')+...
            2*sigma*p*(p-1)*(tmp_fea^(p-2))*(V'*d*X_v)*(X_vv);
    else
        pRhess = @(X,d) hess(X,d)+2*sigma*((d*V)*V')-d.*XGdot;
        phess = @(X,d) hess(X,d)+2*sigma*((d*V)*V');
    end
    
    % if descent direction is unsatisfying, increase tau
    if(success==0)
        tau = min(tau_max,tau*gamma3);
        iter = iter-1;
        success = 1;
    end
    
    % compute descent direction via the second-order method
    subopts.tau = tau; subopts.maxiter = submaxit;
    [Z, subout] = fixpoint_ssn(X,pRgrad,pRhess,alpha,subopts);
    if(subout.success==0||subout.iter==submaxit)
        success = 0;
        continue;
    end
    D = Z-X;
    
    % check stopping criteria
    tol_now = norm(proj_ob(Z)-X,'fro');
    if(tol_now<tol&&iter>1); break; end
    if(iter>=maxiter); break; end
    
    % perform line search
    [Xp, projout] = proj_delta(X,X-pRgrad);
    if(projout.success==0); success = 0; end
    PG = norm(Xp-X,'fro');
    expect_descent = sum(dot(pRgrad,D));
    
    step_s = init_s; tot_sea = 0;
    while(true)
        tot_sea = tot_sea+1;
        X_t = proj_ob(X+step_s*D);
        Dt = X_t-X;
        m_y = sum(dot(pgrad,Dt))+sum(dot(Dt,phess(X,Dt)))/2+...
            tau*norm(Dt,'fro')^2/2;
        if(m_y<-m_a/(m_b+tau)*PG^2)
            break;
        else
            % if cannot find satisfying direction, increase tau
            if(step_s<ac_step)
                success = 0;
                break;
            else
                step_s = step_s/2;
            end
        end
    end
    
    % calculate criterion rho accept X if rho>eta1
    X_v_t = X_t*V;
    feasi_t = norm(X_v_t)^2;
    f_val_t = fval(X_t)+sigma*(feasi_t-1+eps)^p;
    rho = (f_val_t-pfval)/m_y;
    if(rho>eta1)
        X = X_t; X_v = X_v_t; X_vv = X_v*V';
        feasi = feasi_t; pfval = f_val_t;
    end
    
    % adaptively update regularization parameter tau
    if(rho>eta2)
        tau = max(tau_min,tau*gamma1);
    elseif(rho>eta1)
        tau = max(tau_min,min(tau_max,tau*gamma2));
    else
        tau = min(tau_max,tau*gamma3);
    end
    
    % ---- record ----
    if(record&&mod(iter,itprint)==0)
        if(~isfield(subopts,'record')||subopts.record==1)
            for i=1:printab; fprintf('\t'); end
            fprintf('%s', str_head);
        end
        for i=1:printab; fprintf('\t'); end
        fprintf(str_num,iter,pfval,feasi-1,tau,rho,...
            expect_descent/PG/norm(D,'fro'),subout.iter,tol_now,success);
    end
    
    % record as a file
    if(hasRecordFile&&mod(iter,itprint)==0)
        if(isfield(subopts,'recordFile'))
            for i=1:printab; fprintf(fid, '\t'); end
            fprintf(fid, '%s', str_head);
        end
        for i=1:printab; fprintf(fid, '\t'); end
        fprintf(fid,str_num,iter,pfval,feasi-1,tau,rho,...
            expect_descent/PG/norm(D,'fro'),subout.iter,tol_now,success);
    end
    
end % end outer loop

%--------------------------------------------------------------------------
% store the iter. info.

if(hasRecordFile); fclose(fid); end
out.iter = iter;
out.search = tot_sea;
out.feasi = feasi;
out.tau = tau;
out.success = 1;
out.time = toc(ptimetic);

%--------------------------------------------------------------------------

end
