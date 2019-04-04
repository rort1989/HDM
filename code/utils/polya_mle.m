function [est, iters, diff_curve, flag]  = polya_mle(obs,symmetry,varargin) %,method
% perform maximum likelihood estimation on parameters of Polya
% (Dirichlet-multinomial) distribution given samples of discrete values
% Reference: T. P. Minka: Estimating a Dirichlet distribution, 
%         Tehcnical Report 2000, Section 3.

[Q,N] = size(obs);
sanity = sum(obs,2);
check = find(sanity==0);
flag = 0;
if ~isempty(check)
    obs(check,1) = 1;
    flag = 1;
    warning('existing unobserved state');
end
default_max_iters = 20;
default_thresh = 1e-4;
if symmetry
    default_ini = 1;
else
    default_ini = ones(Q,1);
end
p = inputParser;
addOptional(p,'max_iters',default_max_iters,@isnumeric);
addOptional(p,'thresh',default_thresh,@isnumeric);
addOptional(p,'ini',default_ini,@isnumeric);
p.parse(varargin{:});
max_iters = p.Results.max_iters;
thresh = p.Results.thresh;
ini = p.Results.ini;

est = ini;
est_prev = est;
iters = 0;
diff_curve = zeros(max_iters,1);
if symmetry % scalar estimation
    while iters < max_iters
        % compute gradient
        numer = sum(sum(psi(obs + est_prev))) - N*Q*psi(est_prev); % scalar
        denom = Q*(sum(psi(sum(obs) + Q*est_prev)) - N*psi(Q*est_prev)); % scalar
        % update parameter
        est = est_prev*numer/denom;
        % check convergence
        delta_est = norm(est - est_prev);
        est_avg = norm((est + est_prev)/2);
        est_prev = est;
        iters = iters + 1;
        diff_curve(iters) = delta_est/est_avg;
        if diff_curve(iters) < thresh
            break;
        end
    end
else % vector estimation
    while iters < max_iters
        % compute gradient
        for q = 1:Q
            numer = sum(psi(obs(q,:) + est_prev(q))) - N*psi(est_prev(q)); % scalar
            denom = sum(psi(sum(obs) + sum(est_prev))) - N*psi(sum(est_prev)); % scalar
            % update parameter
            est(q) = est_prev(q)*numer/denom;
        end
        % check convergence
        delta_est = norm(est - est_prev);
        est_avg = norm((est + est_prev)/2);
        est_prev = est;
        iters = iters + 1;
        diff_curve(iters) = delta_est/est_avg;
        if diff_curve(iters) < thresh
            break;
        end
    end
end
diff_curve = diff_curve(1:iters);
