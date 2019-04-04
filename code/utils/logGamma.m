function y = logGamma(X, alpha, beta)
% Compute log pdf of a Gamma distribution.
% (beta^alpha)/Gam(alpha)*X^(alpha-1)e^(-beta*X)
% Input:
%   X: 1 x n data vector, each column sums to one (sum(X,1)==ones(1,n) && X>=0)
%   a: 1 x n alpha parameter of Gamma
%   b: 1 x n beta parameter of Gamma
% Output:
%   y: 1 x n probability density in logrithm scale y=log p(x)

c = alpha.*log(beta) - gammaln(alpha);
g = (alpha-1).*log(X) - beta.*X;

y = c + g; % 1 x n