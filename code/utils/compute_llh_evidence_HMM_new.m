function [loglikelihood, viterbicode] = compute_llh_evidence_HMM_new(dataset_test, params, viterbi)
% compute observation loglikelihood given HMM parameters
%
% input: 
%       dataset_test: 1*N array of cells, where each cell contains training sample stored as D*T matrix. 
%               T is the length of sequence which can vary among different samples
%               D is the dimension of each observation; assume continuous observations
%       params: a struct storing HMM parameters
%       viterbi: 1: perform viterbi decoding to estimate hidden state (default:0)
%
% output
%       loglikelihood: 1*N vector contains loglikelihood for each test sample
%       viterbicode: 1*N array of cells contains the most probable hidden
%                           state associated with each sequence

if nargin < 3
    viterbi = 0;
end
num_samples = length(dataset_test);
loglikelihood = zeros(1,num_samples);
viterbicode = cell(1,num_samples);
prior = params.prior;
transmat = params.transmat;
mu = params.mu;
cov = params.cov;
mixmat = params.mixmat;

for i = 1:num_samples
    data = dataset_test{i}; % data is a feature_num*T matrix
    loglikelihood(i) = mhmm_logprob(data, prior, transmat, mu, cov, mixmat);
    if viterbi
        B = mixgauss_prob(data, mu, cov, mixmat);
        viterbicode{i} = viterbi_path(prior, transmat, B);
    end
end

end
