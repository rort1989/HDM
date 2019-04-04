function [params, llh, llh_trace] = hsmm_em_new(obs, L, prior, transmat, duramat, mu, sigma, hyperparams, varargin)
% function perform ML / MAP estimation of parameters in HSMM using EM
% support Gaussian observations
% The particular variant of HSMM used is explicit-duration HMM
% [params, llh] = hsmm_em(data, prior, transmat, mu, Sigma, varargin)
% varargin: hyperparams, mixmat
% input: data: obs
%        maximum duration: L
%           initial guess of parameters: prior, transmat, duramat, mu, Sigma
%           initial guess of hyperparams: hyperparams
%           varargin: max_iter, thresh, verbose, cov_prior, cov_type, dura_type
%%%%%%%%%% latest update: modify to support missing values in observations
%%%%%%%%%% i.e. the 0 values in the obs
%%%%%%%%%% function affected: hsmm_em_new, mixgauss_prob, hsmm_ess_emis (not completed)

% load optional input and format data
p = inputParser;
default_max_iter = 10;
default_thresh = 1e-4;
default_verbose = true;
default_cov_type = 'full';
default_cov_prior = 0;
default_dura_type = 'Multinomial';
default_clamped = 0;
default_mask_missing = cell(1);
addOptional(p,'max_iter',default_max_iter,@isnumeric);
addOptional(p,'thresh',default_thresh,@isnumeric);
addOptional(p,'verbose',default_verbose,@islogical);
addOptional(p,'cov_type',default_cov_type,@ischar);
addOptional(p,'cov_prior',default_cov_prior,@isnumeric);
addOptional(p,'dura_type',default_dura_type,@ischar);
addOptional(p,'clamped',default_clamped,@isnumeric);
addOptional(p,'mask_missing',default_mask_missing,@iscell);
p.parse(varargin{:});
max_iter = p.Results.max_iter;
thresh = p.Results.thresh;
verbose = p.Results.verbose;
cov_type = p.Results.cov_type;
cov_prior = p.Results.cov_prior;
dura_type = p.Results.dura_type;
clamped = p.Results.clamped;
mask_missing = p.Results.mask_missing;
if length(mask_missing) ~= length(obs) % in this case, use complete data
    mask_missing = cell(length(obs),1);
end

% initialize constants
flag = 0;
iter = 1;
llh_p = -Inf;
llh_n = 0;
llh_trace = zeros(max_iter,1);
numex = length(obs);
[O,Q] = size(mu);
if strcmp(dura_type,'Multinomial')
    duramat_table = duramat; % QxL
elseif strcmp(dura_type,'Poisson')
    temp = duramat(:,ones(1,L)).^repmat(0:L-1,Q,1) .* exp(-duramat(:,ones(1,L))) ./ repmat(cumprod([1 1:L-1]),Q,1);
    temp(isnan(temp)) = 0;
    temp(isinf(temp)) = 0;
    duramat_table = mk_stochastic(temp); % normalize the Poisson probability: QxL
    %%%%%%%%%%%%%% assert this is a valid table
else
    error('Unsupported duration distribution.')
end

% while not converged
while flag ~= 1 && iter <= max_iter
% EM algorithm for hierarchical model
% the dynamic parameters are updated for each individual sequence
% the emission parameters are updated using all sequences
    ini_ess_all = zeros(Q,1);
    tr_ess_all = zeros(Q,Q);
    dr_ess_all = zeros(Q,L);
    N_ess_all = zeros(Q,1);
    N_ess_O_all = zeros(O,Q);
    m_ess_all = zeros(O,Q);
    V_ess_all = zeros(O,O,Q);
    % params for each individual sequence
    prior_all = zeros(Q,numex);
    transmat_all = zeros(Q,Q,numex);
    if strcmp(dura_type,'Multinomial')
        duramat_all = zeros(Q,L,numex);
    elseif strcmp(dura_type,'Poisson')
        duramat_all = zeros(Q,numex);
    end
    if iter > 1
        llh_p = llh_n;
        llh_n = 0;
    end
    for n = 1:numex
        % compute emission probability at each frame        
        if isempty(mask_missing{n}) % complete data
            obslik = mixgauss_prob(obs{n}, mu, sigma);  % watch out for very small obslik
        else
            [obslik, frame_not_missing] = mixgauss_prob_miss(obs{n}, mu, sigma, mask_missing{n});
        end
        % E-step: computing sufficient statistics
        % perform inference by collecting forward (alpha), backward (beta),
        % duration (gamma) messages, marignal llh (P(evidence))
        % forward-backward inference 
        %%%%%%%%%%%%%%%%%%%%%% right now this inference assumes categorical duration, but we use it any way for Poission by normalizing the duramat_table, which is finite length
        [~,~,~,delta,~,~,llh,ini_ess,tr_ess,dr_ess,N_ess] = fwdback_hsmm_new(prior, transmat, duramat_table, obslik); %%%%%%%%%%%%%%% needs change for different duration distribution
        % compute expected sufficient statistics for emission part
        if isempty(mask_missing{n}) % complete data  %%%%%%%%%%%% this masking can be omit since the data is masked in 'feature_scaling.m'
            [m_ess, V_ess] = hsmm_ess_emis(obs{n}, delta);
        else
            temp = obs{n}.*mask_missing{n};            
            [m_ess, V_ess] = hsmm_ess_emis(temp(:,frame_not_missing), delta); %%%%%%%%%%%% gamma
        end
        % M-step: maximize the Q function w.r.t. parameters given hyperparameters
        % update initial, transition, duration, emission parameters
        %[prior, transmat, duramat, mu, Sigma] = hsmm_mstep(ini_ess, tr_ess, dr_ess, N_ess, m_ess, V_ess, hyperparams);
        if strcmp(dura_type,'Multinomial')
            [prior_all(:,n), transmat_all(:,:,n), duramat_all(:,:,n)] = hsmm_mstep_dynm(ini_ess, tr_ess, dr_ess, L, hyperparams, 'Multinomial'); % prior not updated for individual sequence %%%%%%%%%%%%%%% needs change for different duration distribution
        elseif strcmp(dura_type,'Poisson')
            [prior_all(:,n), transmat_all(:,:,n), duramat_all(:,n)] = hsmm_mstep_dynm(ini_ess, tr_ess, dr_ess, L, hyperparams, 'Poisson');
        end
        % accumulate ess across sequences for estimating emission parameters later
        ini_ess_all = ini_ess_all + ini_ess;
        tr_ess_all = tr_ess_all + tr_ess;
        dr_ess_all(:,1:size(dr_ess,2)) = dr_ess_all(:,1:1:size(dr_ess,2)) + dr_ess;
        N_ess_all = N_ess_all + N_ess; % should be QxO instead of Qx1 in order to account for missing data
        if ~isempty(mask_missing{n})
            N_ess_O = mask_missing{n}(:,frame_not_missing)*delta';% OxQ %%%%%%%%%%%%% assert N_ess_O = repmat(N_ess',O,1) if mask_missing{n} = ones(O,T)
            N_ess_O_all = N_ess_O_all + N_ess_O;
        end
        m_ess_all = m_ess_all + m_ess; % OxQ
        V_ess_all = V_ess_all + V_ess; % OxOxQ
        llh_n = llh_n + llh(1);
    end
%     prior_ = ini_ess_all/sum(ini_ess_all);
%     [prior, transmat, duramat] = hsmm_mstep_dynm(ini_ess_all, tr_ess_all, dr_ess_all, hyperparams); % prior not updated for individual sequence
%     if clamped == 0
%         [mu, sigma] = hsmm_mstep_emis(N_ess_all, m_ess_all, V_ess_all, hyperparams);
%         sigma = sigma + cov_prior;
%     end
    % check convergence
    llh_trace(iter) = llh_n;
    if verbose
        fprintf('MAP-EM iteration %d completed, total llh = %f\n',iter,llh_n);
    end
    [f_converge, f_decrease] = em_converged(llh_n, llh_p, thresh);
    flag = f_converge ;%+ f_decrease; % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RISKY CONVERGENCE CRITERIA
    %llh_p = llh_n;
    %llh_n = 0;
    iter = iter + 1;
    if flag ~= 1
        [prior, transmat, duramat] = hsmm_mstep_dynm(ini_ess_all, tr_ess_all, dr_ess_all, L, hyperparams, dura_type); % prior not updated for individual sequence %%%%%%%%%%%%%%% needs change for different duration distribution
        if clamped == 0 % choose whether update emission
            if sum(sum(N_ess_O_all)) == 0 % complete data
                [mu, sigma] = hsmm_mstep_emis(N_ess_all, m_ess_all, V_ess_all, cov_type, hyperparams);
            else % incomplete data
                [mu, sigma] = hsmm_mstep_emis(N_ess_O_all, m_ess_all, V_ess_all, cov_type, hyperparams);
            end
            sigma = sigma + cov_prior;
        end
    end
end% end of while
llh_trace = llh_trace(1:iter-1);
params.prior = prior;
params.transmat = transmat;
params.duramat = duramat;
params.mu = mu;
params.sigma = sigma;
params.prior_all = prior_all;
params.transmat_all = transmat_all;
params.duramat_all = duramat_all;
if iter == 2
    llh = llh_n;
else
    llh = llh_p;
end