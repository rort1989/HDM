function [alpha_log,beta_log,gamma_log,delta,xi_log,eta_log,llh,ini_ess,tr_ess,dr_ess,N_ess] = fwdback_hsmm_new(prior, transmat, duramat, obslik, varargin)
% perform forward and backward type inference on hidden semi-Markov model
% (variant: explicit duration model) 
% work in log domain (except the ones conditoned on X) in order to avoid underflow
% 
% this inference is performed on one sequence at a time
% output:
%   alpha(m,d,t) = P(Z(t) = m, D(t) = d, X(1:t)) % unscaled version
%   beta(m,d,t) = P(X(t+1:T) | Z(t) = m, D(t) = d)   
%   xi(m,n,t) = P(X(1:T), Z(t-1)=m, Z(t)=n), m \neq n
%   eta(m,d,t) = P(X(1:T), Z(t-1) ~= m, Z(t)=m, D(t)=d)
%   gamma(m,t) = P(X(1:T), Z(t) = m) ----------------
%   delta(m,t) = P(Z(t) = m | X(1:T))
%   llh = log P(X(1:T))
%   ini_ess(i) = P(Z(1)=i | X(1:T))
%   tr(i,j,t) = P(Z(t) = i, Z(t+1) = j | X(1:T))
%   tr_ess(i,j) = \sum_t tr(i,j,t) 
%   dr(i,d,t) = P(Z(t) = i, D(t) = 1 | X(1:T)) 
%       or equivalently P(Z(t) = i, D(t-d+1) = d | X(1:T)) 
%   dr_ess(i,d) = \sum_t dr(i,d,t)
%   N_ess(i) = \sum_t P(Z(t) = i | X(1:T)) ----------------

% Reference: Yu and Kobayashi, An Efficient Forward-Backward Algorithm for
% an Explicit-Duration Hidden Markov Model, 2003

% process options
% fwd_only, mixmat, obslik2
p = inputParser;
default_fwd_only = false;
addOptional(p,'fwd_only',default_fwd_only,@islogical);
p.parse(varargin{:});
fwd_only = p.Results.fwd_only;

[~, T] = size(obslik);
[Q, L] = size(duramat); % maximum duration of a state
logobslik = log(obslik);
logprior = log(prior); % prior(prior==0) = prior(prior==0) + eps; 
logtransmat = log(transmat); % transmat(transmat==0) = transmat(transmat==0) + eps; 
logduramat = log(duramat); % assume T > L
if T < L
    logduramat = log(mk_stochastic(duramat(:,1:T)));
    L = T;
end

alpha_log = -Inf*ones(Q,L,T);
alpha_log_ = -Inf*ones(T-1,Q);
beta_log = -Inf*ones(Q,L,T);
xi_log = -Inf*ones(Q,Q,T-1);
eta_log = -Inf*ones(Q,L,T-1);
gamma_log = -Inf*ones(Q,T);
delta = zeros(Q,T);
llh = -Inf*ones(1,T);

%~ turn off warning: be careful about this during training only
% idx = find(sum(obslik)==0, 1);
% if ~isempty(idx)
%     warning('extremely low values of emission probability');
% end

% forward pass
t = 1;
alpha_log(:,:,t) = bsxfun(@plus, logduramat, logprior+logobslik(:,t)); % QxL
alpha_log3_ = -Inf*ones(Q,L,2); % pre-allocation
for t = 2:T
    alpha_log(:,1:L-1,t) = bsxfun(@plus, alpha_log(:,2:L,t-1), logobslik(:,t)); % Qx(L-1), log of first term of eq (1)
    %alpha_log(:,1:L-1,t) = alpha_log(:,2:L,t-1) + logobslik(:,t*ones(1,L-1));
    %error1 = norm(alpha_log(:,1:L-1,t)-alpha_log_test)   
    
    ll = bsxfun(@plus, logtransmat, alpha_log(:,1,t-1));
    %ll = logtransmat + squeeze(alpha_log(:,1,(t-1)*ones(1,Q)));
    %error2 = norm(ll-ll_test)
    
    temp = logsumexp(ll); % 1xQ vector, log of bracket subiterm of second term of eq (2)
    temp(isnan(temp)) = -Inf;
    alpha_log_(t-1,:) = temp;
    
    ll = bsxfun(@plus, logduramat, logobslik(:,t));
    %ll = logduramat + logobslik(:,t*ones(1,L));
    %error3 = norm(ll-ll_test)
    
    alpha_log2_ = repmat(temp',1,L) + ll; % QxL, log of second term of eq (2)    
    alpha_log3_(:,:,1) = alpha_log(:,:,t);
    alpha_log3_(:,:,2) = alpha_log2_;
    %temp = logsumexp(alpha_log3_, 3); % QxL    
    offset = max(alpha_log3_(:,:,1),alpha_log3_(:,:,2));
    temp = log(exp(alpha_log3_(:,:,1)-offset) + exp(alpha_log3_(:,:,2)-offset)) + offset;
    %error4 = norm(temp-temp_test)
    
    temp(isnan(temp)) = -Inf;
    alpha_log(:,:,t) = temp;
end

temp = logsumexp(alpha_log(:,:,t),2); % Qx1 vector
temp(isnan(temp)) = -Inf;
gamma_log(:,t) = temp;
temp = logsumexp(gamma_log(:,t));
temp(isnan(temp)) = -Inf;
llh(t) = temp;

if ~fwd_only
    
    % backward pass
    t = T;
    beta_log(:,:,t) = 0;
    delta(:,t) = exp(gamma_log(:,t) - llh(t));
    
    for t = T-1:-1:1
        beta_log(:,2:L,t) = bsxfun(@plus, beta_log(:,1:L-1,t+1), logobslik(:,t+1)); % QxL, d>1 case        
        temp = logsumexp(logduramat + beta_log(:,:,t+1), 2); % Qx1, bracket term of eq(6)
        temp(isnan(temp)) = -Inf;
        beta_log_ = temp;
        beta_log2_ = beta_log_ + logobslik(:,t+1); % Qx1
        temp = logsumexp(bsxfun(@plus, logtransmat, beta_log2_'), 2); % Qx1, eq(6)
        temp(isnan(temp)) = -Inf;
        beta_log(:,1,t) = temp;

        xi_log_ = bsxfun(@plus,logtransmat, alpha_log(:,1,t));
        xi_log_ = bsxfun(@plus,xi_log_,logobslik(:,t+1)'); % QxQ
        xi_log(:,:,t) = bsxfun(@plus,xi_log_, beta_log_'); % QxQ this xi(:,:,t) is xi(:,:,t+1) of eq. 8, diagonal entries should always be -Inf
        %~ check if diagonal entry of xi is -Inf: PASSED
        eta_log_ = alpha_log_(t,:)' + logobslik(:,t+1); % Qx1
        eta_log(:,:,t) = bsxfun(@plus, logduramat + beta_log(:,:,t+1), eta_log_); % QxL
        temp = logsumexp(xi_log(:,:,t),2); % Qx1
        temp(isnan(temp)) = -Inf;
        gamma_log_ = temp;
        
        temp = logsumexp(xi_log(:,:,t)',2); % Qx1
        temp(isnan(temp)) = -Inf;
        gamma_log2_ = temp;
        
        %temp = logsumexp([gamma_log(:,t+1) gamma_log_],2); % Qx1
        offset = max(gamma_log(:,t+1),gamma_log_);
        temp = log(exp(gamma_log(:,t+1)-offset)+exp(gamma_log_-offset)) + offset;
        %error5 = norm(temp-temp_test)
        
        temp(isnan(temp)) = -Inf;
        gamma_log3_ = temp;
        
        %temp = logsubexpv(gamma_log3_,gamma_log2_)'; % Qx1
        d = min(0,gamma_log2_-gamma_log3_); % assume gamma_log3_ is larger, otherwise something is wrong
        temp = gamma_log3_ + log(1-exp(d));
        %error7 = norm(temp-temp_test)        
        temp(isnan(temp)) = -Inf;
        gamma_log(:,t) = temp;
        
        %llh = logsumexp(gamma_log(:,t));
        offset = max(gamma_log(:,t));
        llh(t) = log(sum(exp(gamma_log(:,t)-offset))) + offset;
        %error6 = abs(llh(t)-llh_test)
        
        delta(:,t) = exp(gamma_log(:,t) - llh(t));
    end
    %~ check if llh(1:T) is identical: PASSED    
    ini_ess = exp(gamma_log(:,1) - llh(1));
    tr_ess = exp(logsumexp(xi_log,3) - llh(1));
    tr_ess(isnan(tr_ess)) = 0; % take care the diagonal entries
    dr_ess = exp(logsumexp(eta_log,3) - llh(1));
    dr_ess(isnan(dr_ess)) = 0; % take care never seen length
    N_ess = exp(logsumexp(gamma_log,2) - llh(1));
    N_ess(isnan(N_ess)) = 0;

else
    
    llh = llh(T);
    ini_ess = [];
    tr_ess = [];
    dr_ess = [];
    N_ess = [];
    
end

