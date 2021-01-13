function [Z, out]=fixpoint_ssn(X, Rgrad, Rhess, alpha, opts, varargin)
%--------------------------------------------------------------------------
% Solve the non-linear equation of the form
%
%      F(Z) := Z-\Proj_{\Delta(X)}(Z-alpha(Rgrad+Rhess(Z-X))) = 0
%
% by semi-smooth newton method.
%
% Input:
%    X Rgrad Rhess and alpha --- the non-linear equation            
%    opts --- Options structure with fields
%             tau: regularization term for the non-linear equation  
%             lambda: regularization term for the semi-smooth newton method   
%             lam_min lam_max: value range of lambda
%             eta1 eta2 gamm1 gamma2 gamma3: parameters for updating lambda
%             v_succ: switch control for the semi-smooth method
%             bfix: stepsize of the fixed-point iteration
%             tol_min tol_se: stop control
%             maxiter: max number of iterations      
%             submaxit: max number of iterations for the cg method
%             tau_cg: stop control for the cg method
%             cgsol: cg solver (cg_ssm/bicg/ssm)
%             record: = 0, no print out       
%
% Output:
%         Z --- Solution
%       out --- Output information
%--------------------------------------------------------------------------
% Reference:
% B. Jiang, X. Meng, Z. Wen and X. Chen
% An Exact Penalty Approach For Optimization With Nonnegative Orthogonality
% Constraints
%
% X. Xiao, Y. Li, Z. Wen, and L. Zhang, 
% A Regularized Semi-smooth Newton method with Projection steps for 
% Composite Convex Programs
%
% Author: X. Meng, B. Jiang
% Version 1.0 .... 2021/1

%--------------------------------------------------------------------------

if nargin < 4
    error('at least four inputs: fixpoint_ssn(X,Rgrad,Rhess,alpha)');
elseif nargin < 5
    opts = [];
end

%--------------------------------------------------------------------------
% options for solving the non-linear equation

if ~isfield(opts,'tau');             opts.tau = 5e-2; end
if ~isfield(opts,'bfix');            opts.bfix = 1; end
if ~isfield(opts,'v_succ');          opts.v_succ = 1e2; end

if ~isfield(opts,'lambda');          opts.lambda = 1e1; end
if ~isfield(opts,'lam_min');         opts.lam_min = 1e-3; end
if ~isfield(opts,'lam_max');         opts.lam_max = 1e3; end

if ~isfield(opts,'eta1');            opts.eta1 = 1e-2; end
if ~isfield(opts,'eta2');            opts.eta2 = 1; end
if ~isfield(opts,'gamma1');          opts.gamma1 = 0.7; end
if ~isfield(opts,'gamma2');          opts.gamma2 = 1; end
if ~isfield(opts,'gamma3');          opts.gamma3 = 1.6; end

if ~isfield(opts,'tolse');           opts.tolse = 1e-3; end
if ~isfield(opts,'tol_min');         opts.tol_min = 1e-12; end
if ~isfield(opts,'maxiter');         opts.maxiter = 20; end
if ~isfield(opts,'maxcgit');         opts.maxcgit = 10; end
if ~isfield(opts,'tau_cg');          opts.tau_cg = 5e-2; end
if ~isfield(opts,'cgsol');           opts.cgsol = @cg_ssm; end

if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
else; hasRecordFile = 0;
end
if ~isfield(opts,'record');          opts.record = 0; end
if ~isfield(opts,'itprint');         opts.itprint = 1; end
if ~isfield(opts,'printab');         opts.printab = 3; end

%--------------------------------------------------------------------------
% copy parameters

tau = opts.tau;         bfix = opts.bfix;       v_succ = opts.v_succ;
eta1 = opts.eta1;       eta2 = opts.eta2;       gamma1 = opts.gamma1;
gamma2 = opts.gamma2;   gamma3 = opts.gamma3;   tau_cg = opts.tau_cg;
lambda = opts.lambda;   lam_min = opts.lam_min; lam_max = opts.lam_max;
tolse = opts.tolse;     tol_min = opts.tol_min; maxiter = opts.maxiter;
maxcgit = opts.maxcgit; record = opts.record;   itprint = opts.itprint;
printab = opts.printab; lineq_cg = opts.cgsol;

%--------------------------------------------------------------------------
% prepare for recording iter. info.

stra2 = ['%7s','%10s','%10s','%10s','%14s',' %3s','\n'];
str_head = sprintf(stra2, ...
    'iter', 'F_norm', 'rho', ...
    'lambda','CG(iter/res)','type');
str_num = ['%4d    %+3.2e  %+2.1e  %+2.1e ' ...
    '  %2d %2.1e  %2s\n'];

if(record)
    for i=1:printab; fprintf('\t'); end
    fprintf('Using semi-smooth newton to compute descent direction\n');
    for i=1:printab; fprintf('\t'); end
    fprintf('%s', str_head);
end

% record iter. info. as a file
if(hasRecordFile)
    for i=1:printab; fprintf(fid,'\t'); end
    fprintf(fid,'Using semi-smooth newton to compute descent direction\n');
    for i=1:printab; fprintf(fid, '\t'); end
    fprintf(fid, '%s', str_head);
end

%--------------------------------------------------------------------------
% initial setup

iter = 0;
stimetic = tic;
Z = proj_ob(X);
C = Z-alpha*(Rgrad+Rhess(X,Z-X))+tau*(Z-X);
[C_proj, projout] = proj_delta(X,C);
if(projout.success==0); out.success = 0; return; end

% compute F(Z) and its norm
F = Z-C_proj;
F_norm = norm(F,'fro');
eps = F_norm;
F_start = F_norm;
tot_cg = 0;
out.success = 1;

% main loop
while(true)
    iter = iter+1;
    
    
    % compute direction d by solving linear equation via CG
    Sigma_c = C_proj>0;
    mu = lambda*F_norm;
    
    % the target matrix (not necessarily semi-definite)
    Jacobi_F = @(D_0) (1+mu)*D_0-HSjacobi_projdelta(X,Sigma_c,(1-alpha*tau)*D_0-alpha*Rhess(X,D_0));
    [d, cgout] = lineq_cg(Jacobi_F,-F,tau_cg,maxcgit);
    if(cgout.neg==1); out.success = 0; break; end
    tot_cg = tot_cg+cgout.iter;
    
    
    % compute F(Z+d) and the criterion rho
    d_norm = norm(d,'fro');
    U = Z+d;
    C_U = U-alpha*(Rgrad+Rhess(X,U-X)+tau*(U-X));
    [C_Uproj,projout] = proj_delta(X,C_U);
    if(projout.success==0); out.success = 0; break; end
    F_U = U-C_Uproj;
    F_Unorm = norm(F_U,'fro');
    dot_ud = sum(dot(F_U,d));
    rho = -dot_ud/d_norm^2;
    
    if(F_Unorm<v_succ*eps)
        % accept Z+d if ||F(Z+d)||/||F(Z)|| is not too big
        % note that the semi-smooth newton is not a decreasing method
        Z = U;
        C_proj = C_Uproj;
        F = F_U;
        F_norm = F_Unorm;
        eps = F_Unorm;
        ddflag = 'n';
    else
        % otherwise try a projection step
        v = Z+(dot_ud/F_Unorm^2)*F_U;
        C_v = v-alpha*(Rgrad+Rhess(X,v-X)+tau*(v-X));
        C_vproj = proj_delta(X,C_v);
        F_v = v-C_vproj;
        F_vnorm = norm(F_v,'fro');
        
        if(F_vnorm<F_norm)
            % accept projection if ||F(v)||<||F(Z)||
            Z = v;
            C_proj = C_vproj;
            F = F_v;
            F_norm = F_vnorm;
            ddflag = 'p';
        else
            % otherwise perform the fixed-point iteration
            Z = Z-bfix*F;
            C = Z-alpha*(Rgrad+Rhess(X,Z-X)+tau*(Z-X));
            C_proj = proj_delta(X,C);
            F = Z-C_proj;
            F_norm = norm(F,'fro');
            ddflag = 'f';
        end
    end
    
    % adaptively update regularization parameter lambda
    if(rho>eta2)
        lambda = max(lam_min,lambda*gamma1);
    elseif(rho>eta1)
        lambda = min(lam_max,max(lam_min,lambda*gamma2));
    else
        lambda = min(lam_max,lambda*gamma3);
    end
    
    % check stopping criteria
    if(F_norm<max(tol_min,tolse*F_start)); break; end
    if(iter>=maxiter); break; end
    
    
    % ---- record ----
    if(record&&mod(iter,itprint)==0)
        for i=1:printab; fprintf('\t'); end
        fprintf(str_num,iter,F_norm,rho,lambda,cgout.iter,cgout.res,ddflag);
    end
    
    % record as a file
    if(hasRecordFile&&mod(iter,itprint)==0)
        for i=1:printab; fprintf(fid, '\t'); end
        fprintf(fid,str_num,iter,F_norm,rho,lambda,cgout.iter,cgout.res,...
            ddflag);
    end
end

%--------------------------------------------------------------------------
% store the iter. info.

if(hasRecordFile); fclose(fid); end
out.resi = F_norm;
out.iter = iter;
out.cg = tot_cg;
out.ratio = F_norm/F_start;
out.time = toc(stimetic);

%--------------------------------------------------------------------------

end