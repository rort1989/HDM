function [params, diff_curve] = gamma_mle_new(x, max_iters, thresh)
% compute MLE estimate for gamma distribution given discrete observation
% p(x|a,b) = Gamma(x; a,b) = x^{a-1}/Gamm(a)/b^a exp(-x/b)
% output: params = [a,1/b]
% Reference: Minka 2002 Estimating a Gamma distribution

if nargin < 2
    max_iters = 5;
end
if nargin < 3
    thresh = 1e-3;
end

x_bar = mean(x);
log_x_bar = log(x_bar); % \log(\bar{x})
idx = x>0;
logx_bar = mean(log(x(idx))); % \bar{\log x}
% initialize guess of a
a = 0.5/(log_x_bar - logx_bar);
a_prev = a;
diff_curve = zeros(max_iters,1);
iters = 0;
while iters < max_iters
    numer = logx_bar - log_x_bar + log(a) - psi(a);
    denom = a - a^2*trigamma(a);%psi(a)*gamma(a);%
    ainv = 1/a + numer/denom;
    a = 1/ainv;
    % check convergence
    delta_est = abs(a - a_prev); % (iters)
    est_avg = norm((a + a_prev)/2);
    a_prev = a;
    iters = iters + 1;
    diff_curve(iters) = delta_est/est_avg;
    if diff_curve(iters) < thresh  % (iters-1)
        %delta_est = delta_est(1:iters-1);
        break;
    end
end
diff_curve = diff_curve(1:iters);
b = x_bar/a;
params = [a 1/b];