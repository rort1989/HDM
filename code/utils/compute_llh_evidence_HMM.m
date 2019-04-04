function [loglikelihood, viterbicode] = compute_llh_evidence_HMM(dataset, params, discrete, varargin)
% compute observation loglikelihood given HMM parameters
% [loglikelihood, viterbicode] = compute_llh_evidence_HMM(dataset, params, discrete, varargin)
% dependency: Kevin Murphy's BNT toolbox
%
% input: 
%       dataset: length N array of cells, where each cell contains training sample stored as D*T matrix. 
%               or D*T*N matrix if all instances have continuous-valued observation with same length
%               or N*T matrix if all instances have discrete-valued observation with same length
%               T is the length of sequence which can vary among different samples
%               D is the dimension of each observation; assume continuous observations
%       params: a struct storing HMM parameters
%       discrete: 1: observed node is discrete; 0: observed node is continuous
%       varargin: optional input arguments including
%              'viterbi' - perform viterbi decoding to estimate hidden state (default:0)
%              'llh_sum' - compute loglikelihood of all the samples, output
%              loglikelihood scalar (default:0)
%              'scale' - divide the loglikelihood value of each sequence
%              by its length if not computing llh_sum (default:0)
%
% output:
%       loglikelihood: length N vector contains loglikelihood for each test sample
%       viterbicode: length N array of cells contains the most probable hidden
%                           state associated with each sequence
%
% Author: Rui Zhao
% Date: 2018.01.08

% load optional input
p = inputParser;
default_viterbi = 0;
default_llh_sum = 0;
default_scale = 0;
addOptional(p,'viterbi',default_viterbi,@isnumeric);
addOptional(p,'llh_sum',default_llh_sum,@isnumeric);
addOptional(p,'scale',default_scale,@isnumeric);
p.parse(varargin{:});
viterbi = p.Results.viterbi;
llh_sum = p.Results.llh_sum;
scale = p.Results.scale;

% format dataset
if ~iscell(dataset)
    if discrete == 0
        datacells = num2cell(dataset, [1 2]);
    else
        datacells = num2cell(dataset, 2);
    end
else
    datacells = dataset;
end
ss = size(datacells);
num_samples = max(ss);
loglikelihood = zeros(num_samples,1);
viterbicode = [];
prior = params.prior;
transmat = params.transmat;
if discrete == 0
    mu = params.mu;
    sigma = params.sigma;    
    if isfield(params,'mixmat')
        mixmat = params.mixmat;
    else
        mixmat = ones(size(mu,2),1);
    end
else 
    obsmat = params.obsmat;
end
    
if llh_sum == 1 % compute total llh of all samples
    if discrete == 0
        loglikelihood = mhmm_logprob(datacells, prior, transmat, mu, sigma, mixmat);
    else
        loglikelihood = dhmm_logprob(datacells, prior, transmat, obsmat);
    end
else
    for i = 1:num_samples % compute llh for each instance separately    
        if discrete == 0
            loglikelihood(i) = mhmm_logprob(datacells(i), prior, transmat, mu, sigma, mixmat);
        else
            loglikelihood(i) = dhmm_logprob(datacells(i), prior, transmat, obsmat);
        end
        if scale == 1
            loglikelihood(i) = loglikelihood(i)/numel(datacells{i});
        end
    end
end
    
if viterbi
    viterbicode = cell(ss);
    for i = 1:num_samples % compute llh for each instance separately        
        data = datacells{i}; % data is a feature_num*T matrix
        if discrete == 0
            B = mixgauss_prob(data, mu, sigma, mixmat);
        else
            B = multinomial_prob(data, obsmat);
        end
        viterbicode{i} = viterbi_path(prior, transmat, B);
    end
end

end
