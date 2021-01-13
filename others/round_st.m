function [X, can_round] = round_st(X)
%--------------------------------------------------------------------------
% projection onto the stiefel manifold with nonnegative contraints
%
% can_round: whether X can be rounded to a feasible point

% choose the largest element in each row
[~, wh] = max(X,[],2);
[n,k] = size(X);
if(length(unique(wh))<k)
    can_round = false;
else
    can_round = true;
    X_round = zeros(n,k);
    for i=1:n
        X_round(i,wh(i)) = X(i,wh(i));
    end
    
    % normalization
    X_norm = sqrt(sum(X_round.*X_round));
    X_round = X_round./X_norm;
    X = X_round;
end