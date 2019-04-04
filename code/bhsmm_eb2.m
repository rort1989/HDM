function [hyperparams, params, LLtrace, llhptrace_verbose, hyperparams_trace, llhptrace, llhhtrace] = bhsmm_eb2(data,Q,L,hyperparams,varargin)
% estimate hyperparameters using empirical Bayes approach 
% input data, hyperparameters; difference with eb1: using Poisson duration,
% also compute the likelihood of hyperparameters evaluated on parameters
% P(theta|alpha)
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
default_max_iter_em = 10;
default_cov_type = 'full';
default_cov_prior = 0;
default_dura_prior = 0;
default_dura_type = 'Multinomial';
default_hstate = cell(1);
default_useprior = 0;
default_tol = 1e-2;
default_mask_missing = cell(1);
addOptional(p,'max_iter',default_max_iter,@isnumeric);
addOptional(p,'max_iter_em',default_max_iter_em,@isnumeric);
addOptional(p,'cov_type',default_cov_type,@ischar);
addOptional(p,'cov_prior',default_cov_prior,@isnumeric);
addOptional(p,'dura_prior',default_dura_prior,@isnumeric);
addOptional(p,'dura_type',default_dura_type,@ischar);
addOptional(p,'hstate',default_hstate,@iscell);
addOptional(p,'useprior',default_useprior,@isnumeric);
addOptional(p,'tol',default_tol,@isnumeric);
addOptional(p,'mask_missing',default_mask_missing,@iscell);
p.parse(varargin{:});
max_iter = p.Results.max_iter;
max_iter_em = p.Results.max_iter_em;
cov_type = p.Results.cov_type;
cov_prior = p.Results.cov_prior;
dura_prior = p.Results.dura_prior;
dura_type = p.Results.dura_type;
hstate = p.Results.hstate;
useprior = p.Results.useprior;
tol = p.Results.tol;
mask_missing = p.Results.mask_missing;
N = length(data);
if length(mask_missing) ~= N % in this case, use complete data
    mask_missing = cell(N,1);
end

iter = 1;
previous_llh_total = -1e100;
LLtrace = zeros(1,max_iter);
llhptrace = zeros(1,max_iter);
llhhtrace = zeros(1,max_iter);
llhptrace_verbose = cell(1,max_iter);
%~ for debugging, store values of hyperparameters at each iteration of
%update of hyperparameters
hyperparams_trace = repmat(struct('hyperparams',[]),1,max_iter+1);
hyperparams_trace(1).hyperparams = hyperparams;
% initialize parameters: this must be down together due to the ordering of
% state must be consistent across sequences %%%%%%%%%%%%%%%%%%%%%%%
[prior, transmat, duramat, prior_all, transmat_all, duramat_all, mu, sigma] = hsmm_param_ini(data, Q, size(data{1},1), L, 'cov_type', cov_type, 'cov_prior', cov_prior, 'dura_prior', dura_prior, 'hstate', hstate, 'mask_missing', mask_missing, 'dura_type', dura_type);
trans_prior = mk_stochastic(hyperparams.trans);
% iterate between following two steps while convergence criteria is not met
while iter <= max_iter
% step 1: MAP estimate for parameter of each sequence given current estimate of hyperparameters
    % MAP-EM
    % prior_all, transmat_all, mu, sigma, hyperparams
    [params, llh, llh_trace] = hsmm_em_new(data, L, prior, transmat, duramat, mu, sigma, hyperparams, 'cov_type', cov_type, 'cov_prior', cov_prior, 'max_iter', max_iter_em, 'mask_missing', mask_missing, 'dura_type', dura_type); % choose how many times used to estimate params
    % store the re-estimated params all sequence
    prior = params.prior;
    transmat = params.transmat;
    duramat = params.duramat; % depending on dura_type, this could be QxL (Multinomial) or Qx1 (Poisson), specifying the dura distribution parameters
    mu = params.mu;
    sigma = params.sigma;
    prior_all = params.prior_all;
    transmat_all = params.transmat_all; % this will always be QxLxN as the count of duration values
    duramat_all = params.duramat_all;
    %%%%%%%%%%%% check convergence of estimation
    % 1) llh change; 2) hyperparameter value change
    % display progress    
    llhptrace(iter) = llh; % a summation of llh of each individual sequence
    llhptrace_verbose{iter} = llh_trace; % trace of llh in MAP-EM iterations
    fprintf('Learning iteration %d completed\n',iter);
    iter = iter + 1;
% step 2: ML estimate of hyperparameters of all sequences given current estimate of all parameters
    if useprior > 1
        if useprior < 4 % useprior = 4 case will only execute bhsmm once due to unchanged trans value
        % ML over initial state
        %hyperparams.init = dirichlet_mle(prior_all); % optional input: max_iters,thresh,ini_opt
        transmat_all_mat = zeros(Q-1,Q*N);
        for q = 1:Q
            % ML over transition
            %[hyperparams.trans(q,:)] = dirichlet_mle(squeeze(transmat_all(q,:,:)),30,1e-3);
            idx = setdiff(1:Q,q);
            if length(idx) > 1
                trans = squeeze(transmat_all(q,idx,:)); % (Q-1) x N
                if max(std(trans,[],2)) > 1e-1
                    temp = dirichlet_mle(trans,30,1e-3);  %,iters_tran(q),diff_curve_tran%%%%%%%%%%%% consider change max_iters or thresh
                    if sum(isnan(temp)) == 0
                        hyperparams.trans(q,idx) = temp;
                    end
                end
                transmat_all_mat(:,(q-1)*N+1:q*N) = trans;
            end
            % ML over duration
            if strcmp(dura_type,'Multinomial')
                [hyperparams.dura(q,:)] = dirichlet_mle(squeeze(duramat_all(q,:,:)),30,1e-3);
            elseif strcmp(dura_type,'Poisson')
                dd = squeeze(duramat_all(q,:));
                if std(dd) > 1
                    [hyperparams.dura(q,:)] = gamma_mle_new(dd,10,1e-3);
                end
            else
                error('Unsupported duration distribution.')
            end
            % ML over emission if necessary
        end
        % estimate symmetric transition hyperparameters using transmat_all
        % from all states
        hyperparams.trans_sym = dirichlet_sym_mle(transmat_all_mat, 0, 1e-4); 
        end
        %~ ML over emission 
        [hyperparams.emis.mu, hyperparams.emis.S] = mixgauss_init(1, mu, 'diag');         
        %~
        % likelihood of hyperparameters
        llhhtrace(iter) = llh_hyperparams_hsmm(params,hyperparams,dura_type); % ,llhh_prior(iter-1),llhh_trans(iter-1),llhh_dura(iter-1),llhh_emis_mu(iter-1),llhh_emis_sigma(iter-1)]
        % check convergence of hyperparameter estimation
        hyperparams_trace(iter).hyperparams = hyperparams;
%         trans_prior_new = mk_stochastic(hyperparams.trans);
%         dif = norm(trans_prior_new - trans_prior);
%         trans_prior = trans_prior_new;
%         if dif < tol
%             break;
%         end
        LLtrace(iter) = llhhtrace(iter) + llhptrace(iter-1);
        delta_llh_total = abs(LLtrace(iter) - previous_llh_total);
        avg_llh_total = (abs(LLtrace(iter)) + abs(previous_llh_total) + eps)/2;
        previous_llh_total = LLtrace(iter);
        if (delta_llh_total / avg_llh_total) < tol, 
            break; 
        end
    else % only perform parameter estimation by 'hsmm_em' once
        break;
    end    
end
hyperparams_trace = hyperparams_trace(1:iter);
llhptrace_verbose = llhptrace_verbose(1:iter-1);
llhptrace = llhptrace(1:iter-1);
llhhtrace = llhhtrace(2:iter);
LLtrace = LLtrace(2:iter); % loglikelihood + logprior
%~ debug
% figure; plot(llhh_prior); title('prior')
% figure; plot(llhh_trans); title('trans')
% figure; plot(llhh_dura); title('dura')
% figure; plot(llhh_emis_mu); title('mu')
% figure; plot(llhh_emis_sigma); title('sigma')
% close all;