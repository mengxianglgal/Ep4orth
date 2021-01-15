function [X, Y, out] =  Ep4orth_kind(U, X, Y, opts, varargin)
%--------------------------------------------------------------------------
% An exact penalty problem for solving K-indicators model of the form
%
%                  min     \|UY-X\|_F^2
%                  s.t.    X'X=I, X>=0 Y'Y=I, \|X_{i,:}\|=0
%
% The penalty subproblems are solved by projection gradient method with BB
% stepsize
%
% Input:
%         U --- The features matrix extracted from data
%      X, Y --- Initial guess
%      opts --- Options structure with fields
%               tol: stop control for the subproblem
%               V sigma: parameters for the penalty term
%               feasi_fin: stop control for the constraint violation
%               Xstep: number of gradient step w.r.t. X
%               sigma_max tol_min: maximum/minimum constraints
%                       for simga and tol, respectively.
%               omega1 omega2: parameters for adjusting the
%                       penalty parameters
%               maxoit maxiit: max number of outer/inner iterations
%               BB_type: type of BB stepsize (LBB/SBB/ABB)
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

if nargin < 3
    error('at least three inputs: [X, Y, out] =  Ep4orth_kind(U, X, Y)');
elseif nargin < 4
    opts = [];
end

% size of the problem
[n, k] = size(X);

%-------------------------------------------------------------------------
% options for the projection solver

if ~isfield(opts,'sigma');           opts.sigma = 1e1; end
if ~isfield(opts,'sigma_max');       opts.sigma_max = 1e5; end
if ~isfield(opts,'tol');             opts.tol = 5e-2; end
if ~isfield(opts,'tol_min');         opts.tol_min = 1e-7; end
if ~isfield(opts,'feasi_fin');       opts.feasi_fin = 1e-5; end
if ~isfield(opts,'V');               opts.V = ones(k,1)/sqrt(k); end

if ~isfield(opts,'omega1');          opts.omega1 = 0.5; end
if ~isfield(opts,'omega2');          opts.omega2 = 10; end
if ~isfield(opts,'maxoit');          opts.maxoit = 3e2; end
if ~isfield(opts,'maxiit');          opts.maxiit = 5e1; end

if ~isfield(opts,'init_s');          opts.init_s = 1; end
if ~isfield(opts,'BBtype');          opts.BBtype = 'LBB'; end
if ~isfield(opts,'maxBB');           opts.maxBB = inf; end
if ~isfield(opts,'minBB');           opts.minBB = 0; end
if ~isfield(opts,'Xstep');           opts.Xstep = 1; end

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
BBtype = opts.BBtype;   minBB = opts.minBB;     maxBB = opts.maxBB;
V = opts.V;             init_s= opts.init_s;    Xstep = opts.Xstep;

%-------------------------------------------------------------------------
% prepare for recording iter. info.

stra = ['%4s','%15s','%11s','%9s','%12s','\n'];
str_head = sprintf(stra, ...
    'iter','feasi', ...
    'sigma','tol','sub_step');
str_num = '%4d    %+5.4e   %+2.1e  %+2.1e    %5d  \n';

if(record)
    for i=1:printab; fprintf('\t'); end
    fprintf('K-indicators solver started... \n');
    for i=1:printab; fprintf('\t'); end
    fprintf('%s', str_head);
end

% record iter. info. as a file
if(hasRecordFile)
    for i=1:printab; fprintf(fid, '\t'); end
    fprintf(fid, '%s', str_head);
end

%-------------------------------------------------------------------------
% initial setup

timetic = tic;
rounding = min(min(V*V'));
step_BB = init_s;

% outer loop
for out_iter = 1: maxoit
    
    % inner loop
    stimetic = tic;
    for sub_iter = 1:maxiit
        
        % projection gradint with BB step
        for X_iter = 1:Xstep
            X_pre = X;
            X = proj_ob(U*Y-X*(sigma*V)*V'+(1/step_BB*sigma)*X);
            S = X - X_pre; S = S./norm(S,'fro');
            N = S*V; StN = norm(N,'fro')^2;
            % calculate BB stepsize
            switch BBtype
                case 'LBB'
                    step_BB = 1/StN;
                case 'SBB'
                    step_BB = 1
                case 'ABB'
                    if mod(sub_iter,2) == 0
                        step_BB = 1/StN;
                    else
                        step_BB = 1
                    end
            end
            step_BB = min(maxBB,max(minBB,step_BB));
        end
        
        % update Y
        Y_pre = Y;
        Y = proj_st(U'*X);
        
        % check stopping criteria
        if(sub_iter>=3&&norm(X-X_pre,'fro')<tol&&norm(Y-Y_pre,'fro')<tol)
            break;
        end
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
    
    % check stopping criterion
    if(feasi<=feasi_fin); break; end
    
end

% round X to a feasible solution
if(feasi<rounding)
    X = round_st(X);
    out.post = 1;
else
    out.post = 0;
end


%-------------------------------------------------------------------------
% store the iter. info.

if(hasRecordFile); fclose(fid); end
out.iter = out_iter;
out.time = toc(timetic);
if(min(sum(X,2))<1e-12)
    out.degenerate = 1;
else
    out.degenerate = 0;
end

%--------------------------------------------------------------------------

end