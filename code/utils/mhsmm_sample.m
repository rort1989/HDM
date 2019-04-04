function [obs, hidden] = mhsmm_sample(T, numex, initial_prob, transmat, duramat, mu, Sigma, mixmat, variant)
% SAMPLE_MHSMM Generate random sequences from an HSMM with (mixtures of) Gaussian output.
% [obs, hidden] = sample_mhsmm(T, numex, initial_prob, transmat, mu, Sigma, mixmat)
%
% INPUTS:
% T - length of each sequence
% numex - num. sequences
% init_state_prob(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
% duramat(i,[1 2]) = shape (alpha) and rate (beta) parameter of gamma distribution
% mu(:,j,k) = mean of Y(t) given Q(t)=j, M(t)=k
% Sigma(:,:,j,k) = cov. of Y(t) given Q(t)=j, M(t)=k
% mixmat(j,k) = Pr(M(t)=k | Q(t)=j) : set to ones(Q,1) or omit if single mixture
% variant : duration distribution: 1. Gamma, 2. Categorical, 3. Poisson (Default: 1)
%
% OUTPUT:
% obs(:,t,l) = observation vector at time t for sequence l
% hidden(t,l) = the hidden state at time t for sequence l
if nargin < 9
    variant = 1;
end

Q = length(initial_prob);
if nargin < 8, mixmat = ones(Q,1); end
O = size(mu,1);
% hidden = zeros(T, numex);
obs = zeros(O, T, numex);

hidden = mc_sample_dura(initial_prob, transmat, duramat, T, numex, variant);
for i=1:numex
  for t=1:T
    q = hidden(i,t);
    m = sample_discrete(mixmat(q,:), 1, 1);
    obs(:,t,i) =  gaussian_sample(mu(:,q,m), Sigma(:,:,q,m), 1);
  end
end
