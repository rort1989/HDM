function loglikelihood = compute_llh_evidence_HSMM(datacells, params, L, varargin)
% compute observation loglikelihood given HSMM parameters
% dependency: Kevin Murphy's BNT toolbox
%
% input: 
%       dataset: length N array of cells, where each cell contains training sample stored as D*T matrix. 
%               or D*T*N matrix if all instances have continuous-valued observation with same length
%               or N*T matrix if all instances have discrete-valued observation with same length
%               T is the length of sequence which can vary among different samples
%               D is the dimension of each observation; assume continuous observations
%       params: a struct storing HMM parameters
%       L: maximum length of any sequences
%       varargin: optional input arguments including
%              'dura_type' -  (default:'Multinomial')
%              'mask_missing' - (default:cell(length(datacells),1))
%              'scale' - divide the loglikelihood value of each sequence
%              by its number of data points (default:0)
%
% output:
%       loglikelihood: length N vector contains loglikelihood for each test sample
%
% Author: Rui Zhao
% Date: 2018.08.15

% load optional input
p = inputParser;
default_scale = 0;
default_dura_type = 'Multinomial';
default_mask_missing = cell(length(datacells),1);
addOptional(p,'scale',default_scale,@isnumeric);
addOptional(p,'dura_type',default_dura_type,@ischar);
addOptional(p,'mask_missing',default_mask_missing,@iscell);
p.parse(varargin{:});
scale = p.Results.scale;
dura_type = p.Results.dura_type;
mask_missing = p.Results.mask_missing;

% mask_missing = cell(length(datacells),1);
% format dataset: assume dataset is an array of cells
% if ~iscell(dataset)
%     if discrete == 0
%         datacells = num2cell(dataset, [1 2]);
%     else
%         datacells = num2cell(dataset, 2);
%     end
% else
%     datacells = dataset;
% end
ss = size(datacells);
num_samples = max(ss);
loglikelihood = zeros(num_samples,1);

% viterbicode = [];
% if discrete == 0
%     mu = params.mu;
%     sigma = params.sigma;    
%     if isfield(params,'mixmat')
%         mixmat = params.mixmat;
%     else
%         mixmat = ones(size(mu,2),1);
%     end
% else 
%     obsmat = params.obsmat;
% end

if strcmp(dura_type,'Multinomial')
    for n = 1:num_samples % compute llh for each instance separately
        obslik = mixgauss_prob_miss(datacells{n}, params.mu, params.sigma, mask_missing{n});  % watch out for very small obslik
        [~, ~, ~, ~, ~, ~, loglikelihood(n)] = fwdback_hsmm_new(params.prior, params.transmat, params.duramat, obslik, 'fwd_only', true);%_gamma
        if scale == 1
            loglikelihood(n) = loglikelihood(n)/L;%numel(datacells{n});%
        end
    end
elseif strcmp(dura_type,'Poisson')
    Q = length(params.prior);
    temp = params.duramat(:,ones(1,L)).^repmat(0:L-1,Q,1) .* exp(-params.duramat(:,ones(1,L))) ./ repmat(cumprod([1 1:L-1]),Q,1);
    temp(isnan(temp)) = 0;
    temp(isinf(temp)) = 0;
    duramat_table = mk_stochastic(temp);
    for n = 1:num_samples % compute llh for each instance separately
        obslik = mixgauss_prob_miss(datacells{n}, params.mu, params.sigma, mask_missing{n});  % watch out for very small obslik
        [~, ~, ~, ~, ~, ~, loglikelihood(n)] = fwdback_hsmm_new(params.prior, params.transmat, duramat_table, obslik, 'fwd_only', true);%_gamma
        if scale == 1
            loglikelihood(n) = loglikelihood(n)/L;%numel(datacells{n});%
        end
    end
else
    error('Unsupported duration distribution.')
end


% if llh_sum == 1
%     loglikelihood = sum(loglikelihood);
% end

end
