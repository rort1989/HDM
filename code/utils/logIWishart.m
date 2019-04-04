function y = logIWishart(Sigma, Lambda, v)
% Compute log pdf of an Inverse-Wishart distribution.
% Input:
%   Sigma: d x d covariance matrix
%   Lambda: d x d covariance parameter
%   v: degree of freedom
% Output:
%   y: probability density in logrithm scale y=log p(Sigma)
% Adopted from Mo Chen (sth4nth@gmail.com).
d = length(Sigma);
% B = -0.5*v*logdet(W)-0.5*v*d*log(2)-logmvgamma(0.5*v,d);
% y = B+0.5*(v-d-1)*logdet(Sigma)-0.5*trace(W\Sigma);

B = 0.5*v*logdet(Lambda) - 0.5*v*d*log(2) - logMvGamma(0.5*v,d);
y = B - 0.5*(v+d+1)*logdet(Sigma) - 0.5*trace(Lambda/Sigma);


