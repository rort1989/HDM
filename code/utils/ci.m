function [interval, m] = ci(acc,N,gamma)
% compute confidence interval of classification accuracy
% function [m,interval] = ci(acc,N,gamma)
% input: 
%         acc, classification accuracy
%         N, testing set sample size
%         gamma, confidence level between 0 and 1 (default:0.95)
% output:
%         m: center of confidence interval
%         interval: interval value
% Reference: Ron Kohavi, A study of cross-validation and 
% Bootstrap for accuracy estimation and model selection 1995

if nargin < 3
    gamma = 0.95;
end

p = 0.5 + gamma/2;
z = norminv(p);
denom = (N + z^2);
m = (N*acc + 0.5*z^2)/denom;
offset = z*sqrt(N*acc + 0.25*z^2 - N*acc^2)/denom;
interval = [m-offset, m+offset];