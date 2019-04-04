function [params_samples,llh] = sample_params_hsmm(dataset, params, hyperparams, Q, O, L, M, dura_type, burnin) % discrete
% perform Gibbs sampling on Baysian HSMM to obtain samples of parameters (except prior probability and sigma)

if nargin < 8
    dura_type = 'Multinomial';
end
if nargin < 9
    burnin = round(M/2);
end
% dataset must be an array of cells
N = length(dataset); % number of sequences in dataset
params_samples = repmat(struct(''),M,1);
%~ !!!!!!!!!!!!!!!!!! temporary solution 
inv_sigma_prior = inv(hyperparams.emis.S); %0;% %%%%%%%%% assume same cov for all mean vector
%~
inv_sigma = zeros(O,O,Q);
for q = 1:Q
    %inv_sigma_prior(:,:,q) = inv(hyperparams.emis.S); 
    inv_sigma(:,:,q) = inv(params.sigma(:,:,q));
end
%kappa = hyperparams.emis.kappa;
mu0 = hyperparams.emis.mu; %%%%%%%%%% assume same mu for all mean vector
%nu0 = hyperparams.emis.nu;
Qs = [1:Q]';
const1 = repmat(0:L-1,Q,1);
const2 = repmat(cumprod([1 1:L-1]),Q,1);
params_samples(1).params = params;

%~ debug
% use sample from prior as initialization of parameters
m = 1;
for q = 1:Q
    params_samples(m).params.transmat(q,:) = dirichletrnd(hyperparams.trans(q,:));    
    if strcmp(dura_type,'Multinomial')
        params_samples(m).params.duramat(q,1:L) = dirichletrnd(hyperparams.dura(q,1:L));
    end
    % emission mean
    params_samples(m).params.mu(:,q) = chol(hyperparams.emis.S)'*randn(O,1) + mu0;    
end
if strcmp(dura_type,'Poisson')
    params_samples(m).params.duramat = gamrnd(hyperparams.dura(:,1), 1./hyperparams.dura(:,2)); % Qx1
end
llh = zeros(N,M);
% llh(:,1) = compute_llh_evidence_HSMM(dataset, params_samples(1).params, L, cell(N,1), dura_type);
%~

for m = 2:M+burnin
    %%%%%%%%%%%%% may need multiple iterations on this
    params_samples(m).params = params_samples(m-1).params;
    % sample hidden states
    if strcmp(dura_type,'Multinomial')
        duramat_table = params_samples(m).params.duramat; % QxL
    elseif strcmp(dura_type,'Poisson')
        temp = params_samples(m).params.duramat(:,ones(1,L)).^const1 .* exp(-params_samples(m).params.duramat(:,ones(1,L))) ./ const2;
        temp(isnan(temp)) = 0;
        temp(isinf(temp)) = 0;
        duramat_table = mk_stochastic(temp);% normalize the Poisson probability: QxL        
    else
        error('Unsupported duration distribution.')
    end
    %for re = 1:10
    [z,d] = sample_state_hsmm(dataset,params.prior,params_samples(m).params.transmat,duramat_table,params_samples(m).params.mu,params.sigma);
    %end
    % compute statistics
    % collect statistics
    count_ini = zeros(Q,1);
    count_trans = zeros(Q,Q);
    count_dura = zeros(Q,L);
    count_emis = zeros(Q,O); % mean
    count_emis2 = zeros(O,O,Q); % cov
    count_z = zeros(Q,1);    
    for n = 1:N % loop through all sequences
        % initial state
        C_ini = double(z{n}(1)==Qs);
        count_ini = count_ini + C_ini;
        % transition
        [~,C] = est_transmat(z{n},Q);
        count_trans = count_trans + C - diag(diag(C));
        % state count
        count_z = count_z + sum(C,2) + C_ini;
        % duration
        C = est_duramat(z{n},Q,L);
        count_dura = count_dura + C;        
        % emission
        for q = 1:Q            
            vec = dataset{n}(:,z{n}==q);
            count_emis(q,:) = count_emis(q,:) + sum(vec,2)';
            count_emis2(:,:,q) = count_emis2(:,:,q) + vec*vec';
        end
    end
    % sample parameters from posterior
    % initial state and emission cov are not sampled
    % state initial: alpha; hyper: gamma: 1
    % params_samples(m).prior = dirichletrnd(count_ini+hyperparams.init);    
    % state transition and duration:
    %for re = 1%:10 % repeat a few times
    for q = 1:Q
        cc = count_trans(q,:) + hyperparams.trans(q,:);
        if sum(cc) > 0
            params_samples(m).params.transmat(q,:) = dirichletrnd(cc);
        end
        if strcmp(dura_type,'Multinomial')
            dd = count_dura(q,:) + hyperparams.dura(q,1:L);
            if sum(dd) > 0
                params_samples(m).params.duramat(q,1:L) = dirichletrnd(dd);
            end
        end
        % emission mean
        if count_z(q) > 0
            Lambda = inv_sigma_prior + count_z(q)*inv_sigma(:,:,q);
            Sigma = inv(Lambda);
            Mu = Sigma*(inv_sigma(:,:,q)*count_emis(q,:)' + inv_sigma_prior*mu0); % lambda(p,:) = mu_prior %%%%%%%%%%% new thought in 2017
            params_samples(m).params.mu(:,q) = chol(Sigma)'*randn(O,1) + Mu;
        end
    end
    if strcmp(dura_type,'Poisson')
        gg = sum(count_dura,2) > 0;
        params_samples(m).params.duramat(gg) = gamrnd(count_dura(gg,:)*([0:L-1]') + hyperparams.dura(gg,1), 1./(sum(count_dura(gg,:),2) + hyperparams.dura(gg,2))); % Qx1
    end
    %end
    %~ debug: 
    % evaluate likelihood of the training data on sampled parameters
%     llh(:,m) = compute_llh_evidence_HSMM(dataset, params_samples(m).params, L, cell(N,1), dura_type);
    %
end