function test_onmf
%--------------------------------------------------------------------------
% Orthogonal nonnegative matrix factorization (ONMF)
%
%                  min     \|A-XY^T\|_F^2
%                  s.t.    X'X=I, X>=0, Y>=0.
%
% Data available at http://www.cad.zju.edu.cn/home/dengcai/Data/data.html
%
%--------------------------------------------------------------------------
% Reference:
% B. Jiang, X. Meng, Z. Wen and X. Chen
% An Exact Penalty Approach For Optimization With Nonnegative Orthogonality
% Constraints
%
% Author: X. Meng, B. Jiang
% Version 1.0 .... 2021/1

%--------------------------------------------------------------------------

% choose examples
% Problist = [1:9];
Problist = 3;

% whether record iter. info
record = 1;

% save iter. info.
filesrc = strcat(pwd,filesep,'results');
if ~exist(filesrc, 'dir');     mkdir(filesrc);   end
filepath = strcat(filesrc, filesep, 'onmf');
if ~exist(filepath, 'dir');    mkdir(filepath);  end
strnum = '%6s %2d %+5.4e %+5.4e %+5.4e %+5.4e\n';
fprintf('\t\t\t\t  pur\t\t  ent\t\t time\t\tfeasi\n');

for dprob = Problist    
    % get matrix A and ground truth
    switch dprob
        case 1
            data = load('2k2k.mat');
            A = data.fea;
            test_label = 10;
            true_ans = data.gnd+1;
        case 2
            data = load('Yale_32x32.mat');
            A = data.fea;
            test_label = 15;
            true_ans = data.gnd;
        case 3
            tra = load('TDT2-l10.mat');
            A = tra.A;
            true_ans = tra.true_ans;
            test_label = 10;
        case 4
            tra = load('TDT2-l20.mat');
            A = tra.A;
            true_ans = tra.true_ans;
            test_label = 20;
        case 5
            tra = load('TDT2-t10.mat');
            A = tra.A;
            true_ans = tra.true_ans;
            test_label = 10;
        case 6
            tra = load('TDT2-t20.mat');
            A = tra.A;
            true_ans = tra.true_ans;
            test_label = 20;
        case 7
            tra = load('Reu-t10.mat');
            A = tra.A;
            true_ans = tra.true_ans;
            test_label = 10;
        case 8
            tra = load('Reu-t20.mat');
            A = tra.A;
            true_ans = tra.true_ans;
            test_label = 20;
        case 9
            tra = load('News-t5.mat');
            A = tra.A;
            true_ans = tra.true_ans;
            test_label = 5;
    end
    
    % preprocessing
    [n_,m_] = size(A);
    sumA = full(sum(A));
    A2 = zeros(n_,m_);
    couu = 1;
    for i=1:m_
        if(sumA(i)~=0)
            A2(:,couu) = A(:,i);
            couu = couu+1;
        end
    end
    A = A2(:,1:couu-1);
    [n, m] = size(A);
    
    % get the initial point
    ptimetic = tic;
    [X0, ~, ~] = svds(A,test_label);
    for j=1:test_label
        neg = X0(:,j)<0;
        if norm(X0(neg,j),'fro') > norm(X0(~neg,j),'fro')
            X0(:,j) = -X0(:,j);
        end
    end
    X0 = max(X0,0);
    X0_norm = sqrt(sum(X0.*X0));
    X0 = X0./X0_norm;
    pretime = toc(ptimetic);
    
    % set parameters for Ep4orth_onmf
    clear opts
    opts.sigma = 1e-3;
    opts.omega2 = 1.05;
    opts.omega3 = 0.98;
    opts.altmethod = inf;
    opts.subopts.tau_max = 1e5;
    if(record)
        recordname = strcat(filepath,filesep,'Ep4orth_onmf',...
            'case',num2str(dprob),'.txt');
        opts.recordFile = recordname;
    end

    % run Ep4orth_onmf
    [X, Y, out] = Ep4orth_onmf(A, X0, opts);
    
    % record results, check purity and other criteria
    time = out.time+pretime;
    [~, wh] = max(X,[],2);
    Y = zeros(n,test_label);
    for i=1:n
        Y(i,wh(i)) = 1;
    end
    [pur, ent] = comp_pe(Y,true_ans,1e-8);
    feasi = norm(X'*X-eye(test_label),'fro');
    fprintf(strnum, 'case',dprob,pur,ent,time,feasi);
    
end
end

% compute purity and entropy of given classification results
function [pur, ent] = comp_pe(X,true_ans,eps)
[n, k] = size(X);
tes = zeros(k,k);
for i=1:n
    for j=1:k
        if(X(i,j)>eps)
            tes(true_ans(i),j) = tes(true_ans(i),j)+1;
        end
    end
end
pur = sum(max(tes))/n;
tes_log = log2(tes./sum(tes));
tes2 = tes.*tes_log;
for i=1:k
    for j=1:k
        if(tes(i,j)==0)
            tes2(i,j) = 0;
        end
    end
end
ent = (sum(sum(tes2)))/(-n*log2(k));
end