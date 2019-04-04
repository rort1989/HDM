function [est, iters, diff_curve]  = dirichlet_sym_mle(obs,max_iters,thresh) %,ini_opt
% perform maximum likelihood estimation on parameters of symmetric Dirichlet
% distribution given samples of categorical distribution (proportions)
% Reference: T. P. Minka: Estimating a Dirichlet distribution, 
%         Tehcnical Report 2000, Section 2.1, assuming m=1/K

if nargin < 2
    max_iters = 20;
end
if nargin < 3
    thresh = 1e-4;
end

[Q,N] = size(obs); % each column is a probability vector
% est = zeros(Q,1);
iters = 0;
diff_curve = zeros(max_iters,1);
%~ add eps to 0 entry for numerical stability
[~,ii] = max(obs); % 1xN, idx of maximum probability of each observation
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
logp = sum(log(obs),2)/N; % Qx1
% delta_est = zeros(max_iters,1);
% initialization
%est_prev = Q;
est_prev = mle_ini(obs,Q,logp); % Q*1
if isnan(est_prev)>0
    est_prev = Q; 
    disp('warning: initial guess of dirichlet precision is NaN')
end
est = est_prev;
while iters < max_iters
    % fixed point update (Eq.(33))
    reciprocal = 1/est_prev;
    reciprocal = reciprocal - psi(est_prev) + psi(est_prev/Q) - sum(logp)/Q;
    est = 1/reciprocal;
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

function est_ini = mle_ini(obs,Q,logp)
% compute an intial guess of MLE of symmetric Dirichlet distribution precision
Ep = sum(obs,2)/size(obs,2);
if norm(sum(Ep)-1) >= 1e-12
    error('incorrect input of proportions');
end
denom = -log(Q) - sum(logp)/Q;
est_ini = (Q-1)/2/denom;

