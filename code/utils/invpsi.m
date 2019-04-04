function [X, history]=invpsi(Y, iters)
% X = INVPSI(Y)
%
% Inverse digamma (psi) function.  The digamma function is the
% derivative of the log gamma function.  This calculates the value
% X > 0 for a value X such that digamma(X) = Y.
%
% Reference: T. P. Minka: Estimating a Dirichlet distribution, 
%         Tehcnical Report 2012, Appendix C.
%
% optional dependency: lightspeed toolbox

if nargin < 2
    iters = 5;
end
% initialization
history = zeros(iters,1); max_iters = iters;
M = double(Y >= -2.22);
X = M .*(exp(Y) + 0.5) + (1-M) .* -1./(Y-psi(1)); % X = M .*(exp(Y) + 0.5) + (1-M) .* -1./(Y-digamma(1));
% perform iterations
while iters > 0
  X = X - (psi(X)-Y)./psi(1,X);  % X = X - (digamma(X)-Y)./trigamma(X);
  history(max_iters-iters+1) = X;
  iters = iters - 1;
end