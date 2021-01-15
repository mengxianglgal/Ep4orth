function test_kind
%--------------------------------------------------------------------------
% Solving K-indicators model
%
%                  min     \|UY-X\|_F^2
%                  s.t.    X'X=I, X>=0 Y'Y=I, \|X_{i,:}\|=0
%
% U are extracted from the data matrix.
% Data available at https://github.com/yangyuchen0340/Kind
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
% Problist = 1:6;
Problist = 1:6;

% whether record iter. info
record = 1;

% save iter. info.
filesrc = strcat(pwd,filesep,'results');
if ~exist(filesrc, 'dir');     mkdir(filesrc);   end
filepath = strcat(filesrc, filesep, 'kind');
if ~exist(filepath, 'dir');    mkdir(filepath);  end
strnum = '%6s %2d %+5.4e %+5.4e %+5.4e\n';
fprintf('\t\t\t\t  pur\t\t  ent\t\t  time\n');

for dprob = Problist
    
    % get matrix U and ground truth
    switch dprob
        case 1
            train = load('catsndogs.mat');
        case 2
            train = load('CIFAR100_test.mat');
        case 3
            train = load('CIFAR100_train.mat');
        case 4
            train = load('COIL100_vgg19.mat');
        case 5
            train = load('flower.mat');
        case 6
            train = load('ORL.mat');
        case 7
            train = load('omniglot.mat');
        case 8
            train = load('UK.mat');
    end
    gnd = double(train.gnd);
    U = train.U;
    [n, k] = size(U);
    
    % prepare initial point
    X = max(U,0);
    Y = eye(k);
    
    % set parameters for Ep4orth_kind
    clear opts
    opts.init_s = 10;
    opts.maxBB = 10*k;
    if(record)
        recordname = strcat(filepath,filesep,'Ep4orth_kind',...
            'case',num2str(dprob),'.txt');
        opts.recordFile = recordname;
    end
    
    % run Ep4orth_kind
    [X,Y,out] = Ep4orth_kind(U,X,Y,opts);
    %[X,out] = k_ind_BB(U,opts,X,Y);
    out.time = 1;
    
    % record results, check purity and other criteria
    time = out.time;
    [~, idx] = max(X,[],2);
    [pur, ent] = compute_pen(gnd,idx);
    fprintf(strnum, 'case',dprob,pur,ent,time);
    
end
end

% compute purity and entropy of given classification results
function [pur, ent] = compute_pen(L1,L2)
k = length(unique(L1));
tes = zeros(k,k);
n = length(L1);
for i=1:n
    tes(L1(i),L2(i)) = tes(L1(i),L2(i))+1;
end
    tes_log = log2(tes./sum(tes));
    tes2 = tes.*tes_log;
    for i=1:k
        for j=1:k
            if(tes(i,j)==0)
                tes2(i,j) = 0;
            end
        end
    end
    pur = sum(max(tes))/n;
    ent = (sum(sum(tes2)))/(-n*log2(k));
end