function [prior, transmat, duramat, prior_all, transmat_all, duramat_all, mu, Sigma] = hsmm_param_ini(datacells, hstate_num, feature_dim, max_dura, varargin)
% function to perform initial estimation of parameters given data
% currently only support continuous observations

% optional input: mixture, discrete
% format dataset
% assume in format of cells

% load optional input and format data
p = inputParser;
default_replicate = 10;
default_cov_type = 'full';
default_cov_prior = 0;%0.01*eye(feature_dim);
default_dura_prior = 0;
default_dura_type = 'Multinomial';
default_hstate = cell(1);
default_mask_missing = [];
addOptional(p,'replicate',default_replicate,@isnumeric);
addOptional(p,'cov_type',default_cov_type,@ischar);
addOptional(p,'cov_prior',default_cov_prior,@isnumeric);
addOptional(p,'dura_prior',default_dura_prior,@isnumeric);
addOptional(p,'dura_type',default_dura_type,@ischar);
addOptional(p,'hstate',default_hstate,@iscell);
addOptional(p,'mask_missing',default_mask_missing,@iscell);
p.parse(varargin{:});
replicate = p.Results.replicate;
cov_type = p.Results.cov_type;
cov_prior = p.Results.cov_prior;
dura_prior = p.Results.dura_prior;
dura_type = p.Results.dura_type;
hstate = p.Results.hstate;
mask_missing = p.Results.mask_missing;
if length(mask_missing) ~= length(datacells) % in this case, use complete data
    mask_missing = cell(length(datacells),1);
end

% Define RV dimenstion
N = length(datacells);
O = feature_dim;
Q = hstate_num;
L = max_dura;
% M = mixture;

% initialization of parameters
first_slice_idx = zeros(N,1);
crt_first_slice_idx = 1;
allsamples = zeros(O,N*1000);
for i = 1:N
    % identify and only use the complete frames
    if ~isempty(mask_missing{i})
        % complete the sequence using mean value
        Vm = sum(datacells{i}.*mask_missing{i},2)./sum(mask_missing{i},2); % TxO  %%%%% assert Vm = mean(data_train{n},2) for mask_train{n} = ones(O,T)
        datacells{i} = datacells{i} + Vm(:,ones(1,size(datacells{i},2))).*(1-mask_missing{i});
    end
    first_slice_idx(i) = crt_first_slice_idx;
    crt_first_slice_idx = crt_first_slice_idx + size(datacells{i},2);
    allsamples(:,first_slice_idx(i):crt_first_slice_idx-1) = datacells{i};  % D*total_num_slices
end
allsamples = allsamples(:,1:crt_first_slice_idx-1);
if isempty(hstate{1})
    IDX = kmeans(allsamples',Q,'Replicates',replicate);
else
    IDX = [];
    for i = 1:N
        IDX = [IDX; hstate{i}(:)];
    end
end
% initialization
prior_all = zeros(Q,N);
transmat_all = zeros(Q,Q,N);
duramat_all = ones(Q,L,N);
transmat_all_count = zeros(Q,Q);
duramat_all_count = zeros(Q,L);
% shared by all sequences
mu = zeros(O,Q); 
Sigma = zeros(O,O,Q);
for j = 1:Q
    [mu(:,j), Sigma(:,:,j), mix] = mixgauss_init(1, allsamples(:,IDX == j), cov_type);
    if size(cov_prior,3) == 1
        Sigma(:,:,j) = Sigma(:,:,j) + cov_prior; %%%%%%%%%%%% to prevent too concentrated data distribution
    else
        Sigma(:,:,j) = Sigma(:,:,j) + cov_prior(:,:,j);
    end
    [~,p] = chol(Sigma(:,:,j));
    if p>0
        Sigma(:,:,j) = 100*eye(O);
        warning(fprintf('initial covariance of state %d mixture %d is not psd, use default initialization\n',j));
    end
end
% separate for each sequence
% based on IDX of each sequence
for i = 1:N
    if i < N
        IDX_i = IDX(first_slice_idx(i):first_slice_idx(i+1)-1);
    else
        IDX_i = IDX(first_slice_idx(i):crt_first_slice_idx-1);
    end
    prior_all(IDX_i(1),i) = 1;
    [transmat_all(:,:,i), C] = est_transmat(IDX_i,Q);
    transmat_all_count = transmat_all_count + C;
    %~ forbid self-transition for modeling duration
    transmat_all(:,:,i) = transmat_all(:,:,i) - diag(diag(transmat_all(:,:,i)));
    transmat_all(:,:,i) = mk_stochastic(transmat_all(:,:,i));
    % in case there is only self-transition happening in the count
    transmat_all(:,:,i) = transmat_all(:,:,i) - diag(diag(transmat_all(:,:,i)));
    transmat_all(:,:,i) = mk_stochastic(transmat_all(:,:,i));
    %~ need to ensure each sequence contains all different states
    [duramat_all(:,:,i),C] = est_duramat(IDX_i,Q,L);
%     for q = 1:Q
%         duramat_all(q,:,i) = gamma_mle(C(q,:));
%     end
    duramat_all_count = duramat_all_count + C;
end
prior = sum(prior_all,2)/N; % all sequences share the same initial state distribution
transmat_all_count = transmat_all_count - diag(diag(transmat_all_count));
transmat = mk_stochastic(transmat_all_count);
idx = find(diag(transmat) > 0, 1);
if ~isempty(idx)
    warning('absorbing state which does not transit to a different state');
    % forbid self-transition
    transmat = transmat - diag(diag(transmat));
    transmat = mk_stochastic(transmat);
end

% for q = 1:Q
%     if sum(duramat_all_count(q,:)>0)
%         temp = anisotropic_diffusion_filter(duramat_all_count(q,:),0.2,40,3);
%         duramat(q,:) = gamma_mle(temp);
%         dd(q,:) = gamma_mle(duramat_all_count(q,:));
%     else
%         duramat(q,:) = [1 1]; % in case the state never occurs
%     end
% end

if strcmp(dura_type,'Multinomial')
    duramat = mk_stochastic(duramat_all_count + dura_prior); % QxL
elseif strcmp(dura_type,'Poisson')
    duramat = (duramat_all_count*([0:L-1]')+dura_prior(:,1))./(sum(duramat_all_count,2)+dura_prior(:,2)); % Qx1, dura follows Poisson distribution, dura starts from 0, we shift to start from 1
else
    error('Unsupported duration distribution.')
end

end