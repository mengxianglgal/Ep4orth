function [D] = HSjacobi_projdelta(X,Sigma_c,D)
%-------------------------------------------------------------------------
% compute the HS-jacobian of the operator \proj_{\Delta(X)}[D] at C,
% where Sigma_c = C>0

Sigma_X = Sigma_c.*X;
M = dot(Sigma_X,D)./dot(Sigma_X,X);
D = Sigma_c.*D-Sigma_X.*M;

%-------------------------------------------------------------------------

end