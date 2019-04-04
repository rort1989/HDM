function [params_est, params_ini, LLtrace] = learn_params_HMM_BNT_new(dataset_train, hstate_num, feature_dim, varargin)
% parameter learning of HMM model using BNT's HMM toolbox
%
% input: 
%       dataset_train: 1*N array of cells, where each cell contains training sample stored as D*T matrix. 
%               T is the length of sequence which can vary among different samples
%               D is the dimension of each observation; assume continuous observations
%       hstate_num: an integer specifying hidden state number
%       feature_dim: an integer equals D
%       varargin: optional input arguments including
%               'max_iter' - maximum number of iteration used in EM algorithm (default:10)
%               'cov_type' - 'diag': diagonal covariance matrix for emission probability (default)
%                                 'full': full covariance matrix
%               'cov_prior' - prior value added to diagonal entries of covariance matrix (default: 0)
%               'ini' - 0: use random initialization for parameters estimation
%                      1: use k-means based initialization where k is hstate_num (default)
%               'mixture' - an interger specify the number of mixture used in emission probability (default:1)
%               'thresh' - determine the em convergence condition (default:1e-4)               
%
% output:
%       bnet_trained: BNT struct contains learned HMM model parameters
%       LLtrace: loglikelihood of observation of all samples at each iteration
%       params_ini: a struct contains initial parameters of HMM
%       params_est: a struct same as params_ini contains learned parameters of HMM

% specify default values
max_iter = 10;
cov_type = 'diag';
cov_prior = 0;
ini = 1;
mixture = 1;
thresh = 1e-4;
for argidx = 1:2:nargin-3
    switch varargin{argidx}
        case 'max_iter'
            max_iter = varargin{argidx+1};
        case 'cov_type'
            cov_type = varargin{argidx+1};
        case 'cov_prior'
            cov_prior = varargin{argidx+1};
        case 'ini'
            ini = varargin{argidx+1};   
        case 'mixture'
            mixture = varargin{argidx+1};
        case 'thresh'
            thresh = varargin{argidx+1};
    end
end

% Define RV dimenstion
N = length(dataset_train);
O = feature_dim;
Q = hstate_num;
M = mixture;

% initialization of parameters
if ini
    first_slice_idx = zeros(N,1);
    crt_first_slice_idx = 1;
    allsamples = [];
    for i = 1:N
        first_slice_idx(i) = crt_first_slice_idx;
        crt_first_slice_idx = crt_first_slice_idx + size(dataset_train{i},2);
        allsamples = [allsamples dataset_train{i}];  % D*total_num_slices
    end   
    IDX = kmeans(allsamples',Q,'Replicates',10);
    % initialize prior
    mu0 = zeros(O, Q, M);
    cov0 = zeros(O, O, Q, M);
    prior0 = zeros(Q,1);
    transmat0 = est_transmat(IDX);
    mixmat0 = zeros(Q, M);
    for j = 1:Q
        class_idx = find(IDX == j);
        prior0(j) = length(class_idx)/length(IDX);
        [mu, cov, mix] = mixgauss_init(M, allsamples(:,class_idx), cov_type);
        mu0(:,j,:) = reshape(mu,[O,1,M]);
        cov0(:,:,j,:) = reshape(cov,[O,O,1,M]);
        mixmat0(j,:) = mix';
        for m = 1:M
            [~,p] = chol(cov0(:,:,j,m));
            if p>0
                cov0(:,:,j,m) = 100*eye(feature_dim);
                warning(fprintf('initial cov of state %d mixture %d is not psd, use default initialization\n',j,m));
            end
        end
    end
else
    prior0 = normalise(rand(Q,1));
    transmat0 = mk_stochastic(rand(Q,Q));
    mu0 = 10*randn(O, Q, M);
    cov0 = repmat(100*eye(O), [1 1 Q M]);
    mixmat0 = mk_stochastic(rand(Q,M));
end
params_ini.prior0 = prior0; 
params_ini.mu0 = mu0; 
params_ini.cov0 = cov0; 
params_ini.transmat0 = transmat0; 
params_ini.mixmat0 = mixmat0;

% training
[LLtrace, prior, transmat, mu, cov, mixmat] = mhmm_em( ...
    dataset_train, prior0, transmat0, mu0, cov0, mixmat0, 'max_iter', max_iter, 'cov_type', cov_type, 'cov_prior', cov_prior, 'thresh', thresh);
params_est.prior = prior; 
params_est.mu = mu; 
params_est.cov = cov; 
params_est.transmat = transmat; 
params_est.mixmat = mixmat;

end
