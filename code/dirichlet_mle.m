function [est, iters, diff_curve]  = dirichlet_mle(obs,max_iters,thresh) %,ini_opt
% perform maximum likelihood estimation on parameters of Dirichlet
% distribution given samples of categorical distribution (proportions)
% Reference: T. P. Minka: Estimating a Dirichlet distribution, 
%         Tehcnical Report 2000, Section 1.

if nargin < 2
    max_iters = 20;
end
if nargin < 3
    thresh = 1e-4;
end
% if nargin < 4
%     ini_opt = 2;
% end

[Q,N] = size(obs);
est = zeros(Q,1);
iters = 0;
diff_curve = zeros(max_iters,1);
%~
[~,ii] = max(obs); % 1xN
for n = 1:N
    idx0 = obs(:,n)==0;
    if sum(idx0) == 0
        continue;
    else
        obs(idx0,n) = eps;        
        obs(ii(n),n) = obs(ii(n),n)-sum(idx0)*eps;
    end
end
%~
logp = sum(log(obs),2)/N;
% delta_est = zeros(max_iters,1);
% initialization
%est_prev = dirichlet_mle_ini(obs,Q,N,ini_opt); % Q*1
est_prev = mle_ini(obs,N); % Q*1
est_prev(isnan(est_prev)) = 1;
while iters < max_iters
    for q = 1:Q
        Y = psi(sum(est_prev)) + logp(q);
        [est(q), history] = invpsi(Y);
    end
    % check convergence
    delta_est = norm(est - est_prev); % (iters)
    est_avg = norm((est + est_prev)/2);
    est_prev = est;
    iters = iters + 1;
    diff_curve(iters) = delta_est/est_avg;
    if diff_curve(iters) < thresh  % (iters-1)
        %delta_est = delta_est(1:iters-1);
        break;
    end
end
diff_curve = diff_curve(1:iters);

function est_ini = mle_ini(obs,N)
% compute an intial guess of MLE of Dirichlet distribution parameters
Ep = sum(obs,2)/N;
if norm(sum(Ep)-1) >= 1e-12
    error('incorrect input of proportions');
end
Ep2 = sum(obs.^2,2)/N;
Vp = Ep2 - Ep.^2;
idx = find(Ep > eps);
if length(idx) > 1
    idx = idx(1:end-1);
end
temp = Ep(idx).*(1-Ep(idx))./Vp(idx) - 1;
temp = temp(temp>0 & temp <Inf);
logsumalpha = 1/length(temp)*sum(log(temp));
est_ini = Ep*exp(logsumalpha);

