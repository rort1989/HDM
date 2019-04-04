function [mu, sigma] = est_hyper_emis(data,cov_type,nframe_count)

if nargin < 3
    nframe_count = 50000;    
end

O = size(data{1},1);
data_all = zeros(O,nframe_count);
idx = 0;
for i = 1:length(data)
    T = size(data{i},2);
    data_all(:,idx+1:idx+T) = data{i};
    idx = idx+T;
end
data_all = data_all(:,1:idx);
[mu, sigma, weights] = mixgauss_init(1, data_all, cov_type);
% % same as follows when cov_type = 'full'
% mu_ = mean(data_all,2);
% temp = bsxfun(@minus,data_all,mu_);
% sigma_ = (temp*temp')/idx;
