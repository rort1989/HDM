function [params_est, params_ini, LLtrace] = learn_params_HMM_BNT(dataset, hstate_num, feature_dim, discrete, varargin)
% parameter learning of HMM model
% [params_est, params_ini, LLtrace] = learn_params_HMM_BNT(dataset, hstate_num, feature_dim, discrete, varargin)
% dependency: Kevin Murphy's BNT toolbox
%
% input: 
%       dataset_train: length N array of cells, where each cell contains training sample stored as D*T matrix. 
%               T is the length of sequence which can vary among different samples
%               D is the dimension of each observation for continuous
%               observations or cardinality for discrete observations
%       hstate_num: an integer specifying hidden state number
%       feature_dim: an integer equals D
%       discrete: 0: continuous observed node; 1: discrete observed node
%       varargin: optional input arguments including
%               'max_iter' - maximum number of iteration used in EM algorithm (default:10)
%               'cov_type' - 'diag': diagonal covariance matrix for emission probability (default)
%                                 'full': full covariance matrix
%               'cov_prior' - prior value added to diagonal entries of covariance matrix (default:0)
%               'obs_prior' - prior value added to observation matrix for discrete-valued model (default:0)
%               'tran_prior' - prior value added to state transition matrix (default:0)
%               'ini_prior' - prior value added to state initial vector (default:0)
%               'ini' - 0: use random initialization for parameters estimation
%                       1: use k-means based initialization where k is hstate_num (default:1)
%                       2: use user provided initialization given by optional input 'params'
%               'mixture' - an interger specify the number of mixture used in emission probability (default:1)
%               'thresh' - determine the em convergence condition (default:1e-4)    
%               'sorted' - sort the hidden state assignment according to prior
%                              distribution in ascending manner (default:0)
%               'params' - user supplied initialization of parameters (default:[], only effective with 'ini'>1)
%                              a struct containing fields: 'prior','transmat','obsmat','mu','sigma','mixmat'
%
% output:
%       params_est: a struct same as params_ini contains learned parameters of HMM
%       params_ini: a struct contains initial parameters of HMM
%       LLtrace: loglikelihood of observation of all samples at each iteration
%
% Author: Rui Zhao
% Date: 2017.01.09

% load optional input and format data
p = inputParser;
default_max_iter = 10;
default_cov_type = 'diag';
default_cov_prior = 0;
default_obs_prior = 0;
default_tran_prior = 0;
default_ini_prior = 0;
default_ini = 1;
default_mixture = 1;
default_thresh = 1e-4;
default_sorted = 0;
default_params = [];
addOptional(p,'max_iter',default_max_iter,@isnumeric);
addOptional(p,'cov_type',default_cov_type,@ischar);
addOptional(p,'cov_prior',default_cov_prior,@isnumeric);
addOptional(p,'obs_prior',default_obs_prior,@isnumeric);
addOptional(p,'tran_prior',default_tran_prior,@isnumeric);
addOptional(p,'ini_prior',default_ini_prior,@isnumeric);
addOptional(p,'ini',default_ini,@isnumeric);
addOptional(p,'mixture',default_mixture,@isnumeric);
addOptional(p,'thresh',default_thresh,@isnumeric);
addOptional(p,'sorted',default_sorted,@isnumeric);
addOptional(p,'params',default_params,@isstruct);
p.parse(varargin{:});
max_iter = p.Results.max_iter;
cov_type = p.Results.cov_type;
cov_prior = p.Results.cov_prior;
obs_prior = p.Results.obs_prior;
tran_prior = p.Results.tran_prior;
ini_prior = p.Results.ini_prior;
ini = p.Results.ini;
mixture = p.Results.mixture;
thresh = p.Results.thresh;
sorted = p.Results.sorted;
params = p.Results.params;
% format dataset
if ~iscell(dataset)
    if ~discrete
        datacells = num2cell(dataset, [1 2]);
    else
        datacells = num2cell(dataset, 2);
    end
else
    datacells = dataset;
end

% Define RV dimenstion
N = length(datacells);
O = feature_dim;
Q = hstate_num;
M = mixture;

% initialization of parameters
if ini == 1
    first_slice_idx = zeros(N,1);
    crt_first_slice_idx = 1;
    if ~discrete
        allsamples = zeros(O,N*1000);
    else
        allsamples = zeros(1,N*1000);
    end
    for i = 1:N
        first_slice_idx(i) = crt_first_slice_idx;
        crt_first_slice_idx = crt_first_slice_idx + size(datacells{i},2);
        allsamples(:,first_slice_idx(i):crt_first_slice_idx-1) = datacells{i};  % D*total_num_slices
    end
    allsamples = allsamples(:,1:crt_first_slice_idx-1);
    if ~discrete % continuous observed node        
        IDX = kmeans(allsamples',Q,'Replicates',10);
        % initialize prior
        mu0 = zeros(O, Q, M);
        sigma0 = zeros(O, O, Q, M);
        prior0 = zeros(Q,1);
        transmat0 = est_transmat(IDX,Q);
        mixmat0 = zeros(Q, M);
        for j = 1:Q
            class_idx = find(IDX == j);
            class_idx_first = intersect(class_idx,first_slice_idx);
            prior0(j) = length(class_idx_first)/N; % prior0(j) = length(class_idx)/length(IDX);
            [mu, sigma, mix] = mixgauss_init(M, allsamples(:,class_idx), cov_type);
            mu0(:,j,:) = reshape(mu,[O,1,M]);
            sigma0(:,:,j,:) = reshape(sigma,[O,O,1,M]);
            mixmat0(j,:) = mix';
            for m = 1:M
                [~,p] = chol(sigma0(:,:,j,m));
                if p>0
                    sigma0(:,:,j,m) = 100*eye(O);
                    warning(sprintf('initial covariance of state %d mixture %d is not psd, use default initialization\n',j,m));
                end
            end
        end
    else % discrete observed node _____________________ lack better way of initialization comparing to continuous case
        prior0 = normalise(rand(Q,1));
        transmat0 = mk_stochastic(rand(Q,Q));
        temp = zeros(1,O);
        for o = 1:O
             temp(o) = sum(allsamples==o)/length(allsamples);
        end    
        obsmat0 = repmat(temp,Q,1); % size Q*O
    end
    
elseif ini > 1 % user specified initialization
    prior0 = params.prior;
    transmat0 = params.transmat;
    if ~discrete    
        mu0 = params.mu;
        sigma0 = params.sigma;
        if isfield(params,'mixmat')
            mixmat0 = params.mixmat;
        else
            mixmat0 = ones(Q, M);
        end
    else
        obsmat0 = params.obsmat;
    end

else % random initialization
    prior0 = normalise(rand(Q,1));
    transmat0 = mk_stochastic(rand(Q,Q));
    if ~discrete    
        mu0 = 10*randn(O, Q, M);
        sigma0 = repmat(100*eye(O), [1 1 Q M]);
        mixmat0 = mk_stochastic(rand(Q,M));
    else
        obsmat0 = mk_stochastic(rand(Q,O));
    end
end

if sorted
    [prior0,order] = sort(prior0);
    transmat0 = transmat0(order,order);
    if ~discrete
        mu0 = mu0(:,order,:);
        sigma0 = sigma0(:,:,order,:);
        mixmat0 = mixmat0(order,:);
    else
        obsmat0 = obsmat0(order,:);
    end
end

params_ini.prior = prior0; 
params_ini.transmat = transmat0; 
if ~discrete
    params_ini.mu = mu0; 
    params_ini.sigma = sigma0; 
    params_ini.mixmat = mixmat0;
else
    params_ini.obsmat = obsmat0;
end

% training
if ~discrete
    [LLtrace, prior, transmat, mu, sigma, mixmat] = mhmm_em( ...
    datacells, prior0, transmat0, mu0, sigma0, mixmat0, 'max_iter', max_iter, 'cov_type', cov_type, 'cov_prior', cov_prior, 'thresh', thresh);
else
    [LLtrace, prior, transmat, obsmat] = dhmm_em2(datacells, prior0, transmat0, obsmat0, 'max_iter', max_iter, 'thresh', thresh, ...
        'obs_prior_weight', obs_prior, 'tran_prior_weight', tran_prior, 'ini_prior_weight', ini_prior);
end
params_est.prior = prior; 
params_est.transmat = transmat; 
if ~discrete
    params_est.mu = mu; 
    params_est.sigma = sigma;
    params_est.mixmat = mixmat;
else
    params_est.obsmat = obsmat;
end

end
