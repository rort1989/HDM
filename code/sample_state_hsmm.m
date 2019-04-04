function [z, d, data] = sample_state_hsmm(data,prior,transmat,duramat,mu,sigma)
% function to blocked sample hidden state in HSMM (explicit duration HMM) given parameters
% input: data: matrix (each row corresponds to one sequence) or cell array
%          (each cell corresponds to one sequence)
% output: z: a matrix or an array of cells (depending on 'data') corresponding to hidden state sequences of
%         data: same as input 'data' with structure reformed as an array of cells each input sequence

% decide number of sequences 
if iscell(data)
    N = length(data);
else
    %N = max(size(data,1),size(data,3)); % either a 2D matrix for discrete obs or 3D matrix for continuous obs    
    N = size(data,3); % only support continuous data now
    % data = num2cell(data,2);    
    data = num2cell(data,[1 2]);    
end
z = cell(N,1);
d = cell(N,1);
for n = 1:N
    % compute observation likelihood of each frame P(y_t|z_t)
    % likelihood = compute_lhd(data{n},mu,discrete);
    likelihood = mixgauss_prob(data{n}, mu, sigma);
    % compute backward message recursively (Yu 2003, Murphy 2010)
    % [msg, msg_partial] = backward_msg_hsmm(likelihood,transmat,duramat); % similar to the backward part of fwdback_hsmm
    % alpha(m,d,t) = P(Z(t) = m, D(t) = d, X(1:t)) % unscaled version
    % beta(m,d,t) = P(X(t+1:T) | Z(t) = m, D(t) = d)
    % eta(m,d,t) = P(X(1:T), Z(t-1) ~= m, Z(t)=m, D(t)=d) % t=1:T-1
    % gamma(m,t) = P(X(1:T), Z(t) = m)
    [alpha_log,beta_log,gamma_log,~,~,eta_log,llh] = fwdback_hsmm_new(prior, transmat, duramat, likelihood);
    msg_partial = exp(gamma_log-llh(1)); % MxT
    % sample hidden state 
    T = size(data{n},2);
    z_this = zeros(1,T);
    d_this = zeros(1,T);
    
    % sample initial state and duration
    table = prior.*msg_partial(:,1);%exp(gamma_log(:,1)-llh(1))
    table = cumsum(table);
    z_this(1) = 1 + sum(rand*table(end)>table);    
    table = exp(squeeze(alpha_log(z_this(1),:,1) + beta_log(z_this(1),:,1)) - gamma_log(z_this(1),1)); % P(X(1:T),D(1)=d,Z(1)=z_this(1))/P(X(1:T),Z(1)=z_this(1))
    table = cumsum(table);
    d_this(1) = 1 + sum(rand*table(end)>table);
    % sample d_t conditioned on z_t, then sample z_{t+1} again condition on z_{t-1}, recursive
    for t = 2:T
        if d_this(t-1) - 1 > 0 % stay in same state, count down
            z_this(t) = z_this(t-1);
            d_this(t) = d_this(t-1)-1;
        else
            % sample to enter a new state
            table = transmat(z_this(t-1),:)'.*msg_partial(:,t);%exp(gamma_log(:,t)-llh(t)) 
            table = cumsum(table);
            z_this(t) = 1 + sum(rand*table(end)>table);
            table = exp(squeeze(eta_log(z_this(t),:,t-1))-gamma_log(z_this(t),t));
            table = cumsum(table);
            d_this(t) = 1 + sum(rand*table(end)>table);
        end
    end
    z{n} = z_this;
    d{n} = d_this;
end
