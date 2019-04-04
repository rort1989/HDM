function [params_samples_, LLtrace] = bhsmm_fb(data,Q,O,L,MC,varargin)
% generate posterior samples of hyperparameters using full Bayesian approach 
% input data, hyperprior
%%%%%%%%%%%%%%%%%%% used bnt functions:
% 'mixgauss_prob.m'
% 'mk_stochastic.m'
% 'normalise.m'
% 'em_converged.m'
% 'logsumexp.m'
% 'logsumexpv.m'
% 'est_transmat.m'
% 'mixgauss_init.m', 'gmm.m', 'gmminit.m'
%%%%%%%%%%%%%%%%%%% hyperparameters:
% hyperparams.init = ones(Q,1);
% hyperparams.trans = ones(Q,Q);
% hyperparams.dura = ones(Q,1); %%%%%%%%%%%%%%%% use Poisson distribution
% hyperparams.emis.S = zeros(O);
% hyperparams.emis.nu = -1-O;
%%%%%%%%%%%%%%%%%%% parameters:
% prior_all = zeros(Q,num_inst);
% transmat_all = zeros(Q,Q,num_inst);
% duramat_all = zeros(Q,L,num_inst);
% mu = zeros(O,Q);
% sigma = zeros(O,O,Q);
p = inputParser;
default_max_iter = 3;
default_cov_prior = 0;
default_cov_type = 'full';
default_dura_prior = 0;
default_dura_type = 'Multinomial';
default_hstate = cell(1);
default_tol = 1e-2;
default_mask_missing = cell(1);
addOptional(p,'max_iter',default_max_iter,@isnumeric);
addOptional(p,'cov_prior',default_cov_prior,@isnumeric);
addOptional(p,'cov_type',default_cov_type,@ischar);
addOptional(p,'dura_prior',default_dura_prior,@isnumeric);
addOptional(p,'dura_type',default_dura_type,@ischar);
addOptional(p,'hstate',default_hstate,@iscell);
addOptional(p,'tol',default_tol,@isnumeric);
addOptional(p,'mask_missing',default_mask_missing,@iscell);
p.parse(varargin{:});
max_iter = p.Results.max_iter;
cov_type = p.Results.cov_type;
cov_prior = p.Results.cov_prior;
dura_prior = p.Results.dura_prior;
dura_type = p.Results.dura_type;
hstate = p.Results.hstate;
tol = p.Results.tol;
mask_missing = p.Results.mask_missing;
if length(mask_missing) ~= length(data) % in this case, use complete data
    mask_missing = cell(length(data),1);
end

N = length(data);
iter = 1;
LLtrace = zeros(1,max_iter);
hyperparams.init = zeros(Q,1);
hyperparams.trans = zeros(Q,Q); % subtraction for imposing no prior on self-transition
hyperparams.dura = zeros(Q,L); % need this prior to handle non-existing
% hyperparams.emis.kappa = 0;
hyperparams.emis.mu = 0;
hyperparams.emis.S = 0;
% hyperparams.emis.nu = -2-O;
max_llh = -Inf;

% initialize hyperparameters: this must be done together due to the ordering of
% state must be consistent across sequences %%%%%%%%%%%%%%%%%%%%%%%
% hyperparams = hsmm_hyperparam_ini(data);
[params.prior, params.transmat, params.duramat, ~, ~, ~, params.mu, params.sigma] = hsmm_param_ini(data, Q, size(data{1},1), L, 'cov_type', cov_type, 'cov_prior', cov_prior, 'dura_prior', dura_prior, 'hstate', hstate, 'mask_missing', mask_missing, 'dura_type', dura_type);
% iterate between following two steps while convergence criteria is not met
while iter <= max_iter
    % step 1: sample parameter of each sequence given current sample of hyperparameters
    params_samples = sample_params_hsmm(data, params, hyperparams, Q, O, L, MC, dura_type);    
    % step 2: sample hyperparameters of all sequences given current samples of all parameters
    % hyperparams_samples = sample_hypers_hsmm(transmat_all, duramat_all, mu_all, hyperprior, dura_type);
    % step 3: check likelihood once in a while
    % 1) llh change; 2) hyperparameter value change
    llh = zeros(N,MC);
    for i = 1:MC
        llh(:,i) = compute_llh_evidence_HSMM(data,params_samples(i).params,L,mask_missing,dura_type);
    end
    history.llh(iter) = mean(llh(:));
    LLtrace(iter) = history.llh(iter); % a summation of llh of each individual sequence
    fprintf('Inference iteration %d completed\n',iter);
    if LLtrace(iter) > max_llh
        params_samples_ = params_samples;
        max_llh = LLtrace(iter);
    end
    iter = iter + 1;    
end

LLtrace = LLtrace(1:iter-1);