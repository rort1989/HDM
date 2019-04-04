function [total_uncertainty, total_cov_avg, probs, mean_cov, cov_mean, N, covs] = est_uncertainty(loglikelihood_all,nframe_train)
% compute uncertainty estimation based on loglikelihood obtained by
% different samples of parameters of different classes

% compute the total uncertainty of training data
% first term
[N, num_class, MC] = size(loglikelihood_all);
probs = zeros(size(loglikelihood_all)); % N_train x num_class x MC
covs = zeros(num_class,num_class,N,MC); % num_class x num_class x N_train x MC
for m = 1:MC % iterate over each MC sample
    llh_m = loglikelihood_all(:,:,m);
    mask = (~isnan(llh_m)).*(~isinf(llh_m));
    llh_m_scale = bsxfun(@rdivide, llh_m, nframe_train);
    llh_m_scale(~mask) = -Inf;
    temp = logsumexp(llh_m_scale,2);
    llh_offset = llh_m_scale - temp(:,ones(1,num_class));
    probs(:,:,m) = exp(llh_offset);
    for n = 1:N
        covs(:,:,n,m) = diag(probs(n,:,m))-probs(n,:,m)'*probs(n,:,m);
    end
end
mean_cov = mean(covs,4); % num_class x num_class x N_train
% second term: model uncertainty
cov_mean = zeros(num_class,num_class,N);
for n = 1:N
    cov_mean(:,:,n) = cov(squeeze(probs(n,:,:))');
end
% Two terms together: total covariancance
total_cov = mean_cov + cov_mean; % num_class x num_class x N_train
total_cov_avg = mean(total_cov,3);
total_uncertainty = zeros(N,1);
for n = 1:N
    total_uncertainty(n) = trace(total_cov(:,:,n));
end