function [X, out] =  Ep4orth_proj(C, X, opts, varargin)
%--------------------------------------------------------------------------
% An exact penalty problem for solving projection problem onto stiefel
% manifold with non-negative constraints
%
%                  min     \|C-X\|_F^2
%                  s.t.    X'X=I, X>=0
%
% The penalty subproblems are solved by projection gradient method with BB
% stepsize and non-monotone line search
%
% Input:
%         C --- The matrix to be projected on
%         X --- Initial guess
%      opts --- Options structure with fields
%               tol: stop control for the subproblem
%               V sigma: parameters for the penalty term
%               feasi_fin: stop control for the constraint violation
%               sigma_max tol_min: maximum/minimum constraints
%                       for simga and tol, respectively.
%               omega1 omega2: parameters for adjusting the
%                       penalty parameters
%               maxoit maxiit: max number of outer/inner iterations
%               ac_step: minimum acceptable stepsize (for line search)
%               ac_ratio:  minimum acceptable rate of decline
%               BB_type: type of BB stepsize (LBB/SBB/ABB)
%               record: = 0, no print out
%
% Output:
%         X --- Solutions
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
    error('at least two inputs: [X, out] = Ep4orth_proj(C,X)');
elseif nargin < 3
    opts = [];
end

% size of the problem
[n, k] = size(X);

%-------------------------------------------------------------------------
% options for the projection solver

if ~isfield(opts,'sigma');           opts.sigma = 1e-2; end
if ~isfield(opts,'sigma_max');       opts.sigma_max = 1e10; end
if ~isfield(opts,'tol');             opts.tol = 0.1; end
if ~isfield(opts,'tol_min');         opts.tol_min = 1e-7; end
if ~isfield(opts,'feasi_fin');       opts.feasi_fin = 1e-8; end
if ~isfield(opts,'V');               opts.V = ones(k,1)/sqrt(k); end

if ~isfield(opts,'omega1');          opts.omega1 = 0.8; end
if ~isfield(opts,'omega2');          opts.omega2 = 2; end
if ~isfield(opts,'maxoit');          opts.maxoit = 1e2; end
if ~isfield(opts,'maxiit');          opts.maxiit = 1e2; end

if ~isfield(opts,'ac_step');         opts.ac_step = 1e-1; end
if ~isfield(opts,'ac_ratio');        opts.ac_ratio = 1e-4; end
if ~isfield(opts,'init_s');          opts.init_s = 1; end
if ~isfield(opts,'gamma');           opts.gamma = 0.85; end
if ~isfield(opts,'BBtype');          opts.BBtype = 'SBB'; end
if ~isfield(opts,'maxBB');           opts.maxBB = inf; end

if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
else; hasRecordFile = 0; 
end
if ~isfield(opts, 'record');         opts.record = 0; end
if ~isfield(opts,'itprint');         opts.itprint = 1; end
if ~isfield(opts,'printab');         opts.printab = 0; end

%--------------------------------------------------------------------------
% copy parameters

sigma = opts.sigma;     tol = opts.tol;         sigma_max = opts.sigma_max;      
tol_min = opts.tol_min; maxoit = opts.maxoit;   maxiit = opts.maxiit;
omega1 = opts.omega1;   omega2 = opts.omega2;   feasi_fin = opts.feasi_fin; 
record = opts.record;   itprint = opts.itprint; printab = opts.printab;
BBtype = opts.BBtype;   ac_step = opts.ac_step; ac_ratio = opts.ac_ratio;
V = opts.V;             init_s= opts.init_s;    maxBB = opts.maxBB;
gamma = opts.gamma;

%-------------------------------------------------------------------------
% prepare for recording iter. info.

stra = ['%4s','%15s','%11s','%9s','%12s','\n'];
str_head = sprintf(stra, ...
    'iter','feasi', ...
    'sigma','tol','sub_step');
str_num = '%4d     %5.4e    %2.1e    %2.1e     %5d  \n';

if(record)
    for i=1:printab; fprintf('\t'); end
    fprintf('Projection solver started... \n');
    for i=1:printab; fprintf('\t'); end
    fprintf('%s', str_head);
end

% record iter. info. as a file
if(hasRecordFile)
    for i=1:printab; fprintf('\t'); end
    fprintf(fid,'Projection solver started... \n');
    for i=1:printab; fprintf(fid, '\t'); end
    fprintf(fid, '%s', str_head);    
end

%-------------------------------------------------------------------------
% initial setup

timetic = tic;
rounding = min(min(V*V'));
Xinit = X; feasi_pre = Inf;
tot_sea = 0; fea_fail = 0;
step_BB = init_s;

% outer loop
for out_iter = 1: maxoit
    
    % Calculate the value of penal function, reset parameters for line
    % search
    f = -sum(dot(X,C))+ (sigma/2)*norm(X*V,'fro')^2;
    fref = f; Q = 1;
    
    % inner loop
    stimetic = tic;
    for sub_iter = 1:maxiit
        
        % non-monotone line search
        X_pre = X; 
        while(true)
            tot_sea = tot_sea+1;
            X = X_pre+step_BB*(C/sigma-(X_pre*V)*V');
            X = proj_ob(X);
            f = -sum(dot(X,C))+ (sigma/2)*norm(X*V,'fro')^2;
            if(f <= fref-ac_ratio/step_BB*norm(X-X_pre,'fro')^2)
                break;
            elseif(step_BB<ac_step)
                break;
            else
                step_BB = 0.5*step_BB; 
            end
        end
        
        % calculate BB stepsize
        switch BBtype
            case 'LBB'
                S = X - X_pre; S = S./norm(S,'fro');
                N = S*V;
                StN = norm(N,'fro')^2;
                step_BB = min(1/StN,maxBB); 
            case 'SBB'
                step_BB = 0.99; 
            case 'ABB' 
                S = X - X_pre; S = S./norm(S,'fro');
                N = S*V;
                StN = norm(N,'fro')^2;
                if mod(sub_iter,2) == 0
                    step_BB = min(1/StN,maxBB); 
                else
                    step_BB = 0.99;
                end
        end
        
        % check stopping criteria
        if(sub_iter>=3&&norm(X-X_pre,'fro')<tol); break; end 
        
        % update parameters of non-monotone line search
        Qp = Q; Q = gamma*Qp + 1; fref = (gamma*Qp*fref + f)/Q;
    end
    feasi = abs(norm(X*V,'fro')^2 - 1);
    
    % store the iter. info.
    out.subiter(out_iter) = sub_iter;
    out.feasi(out_iter) = feasi;
    out.time(out_iter) = toc(stimetic);
    
    % ---- record ----
    if(record&&mod(out_iter,itprint)==0)
        for i=1:printab; fprintf('\t'); end
        fprintf(str_num,out_iter,feasi,sigma,tol,sub_iter);
    end
    
    % record as a file
    if(hasRecordFile&&mod(out_iter,itprint)==0)
        for i=1:printab; fprintf(fid, '\t'); end
        fprintf(fid,str_num,out_iter,feasi,sigma,tol,sub_iter);
    end
    
    % update parameters
    tol = max(tol_min,tol*omega1);
    sigma = min(sigma_max,sigma*omega2);
    
    % if feasi do not decrease sufficiently, increase fea_fail
    if(feasi>0.95*feasi_pre)
        fea_fail =  fea_fail+1;
    end
    feasi_pre = feasi;
    
    % round X to a feasible point and set it to be the initial point 
    [X_tmp, can_proj] = round_st(X);
    if(can_proj);  Xinit = X_tmp; end
    
    % check stopping criterion
    if(feasi<=feasi_fin); break; end
    
    % if the feasibility does not decrease (stuck at saddle), reset X
    if (fea_fail>=5)
        fea_fail = 0;
        X = Xinit;
    end
    
end

% round X to a feasible solution
if(feasi<rounding);  X = round_st(X); end

% post-processing, solve the projection problem restricted on the supp(X)
Cpost = C.*(X>0);
X = proj_ob(Cpost);

%-------------------------------------------------------------------------
% store the iter. info.

if(hasRecordFile); fclose(fid); end
out.nproj = tot_sea;
out.iter = out_iter;
out.time = toc(timetic);

%--------------------------------------------------------------------------

end