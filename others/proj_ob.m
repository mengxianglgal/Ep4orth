function X = proj_ob(X)
%--------------------------------------------------------------------------
% projection onto the oblique manifold with nonnegative contraints

[~,k] = size(X);
[ma,wh] = max(X);
X = max(X,0);
for i=1:k
    % if max_j X_ji>0, normalize X_{:,i}
    if(ma(i)>0)
        t = X(:,i);
        X(:,i) = t./norm(t);
    % otherwise set argmax_j X_ji to 1 
    else
        X(wh(i),i) = 1;
    end
end

%--------------------------------------------------------------------------

end