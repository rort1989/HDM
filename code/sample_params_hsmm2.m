function [params_samples,llh,params_samples_all,states] = sample_params_hsmm2(dataset, params, hyperparams, Q, O, L, M, dura_type, burnin, llh_flag, weight) % discrete
% perform Gibbs sampling on Baysian HSMM to obtain samples of parameters (except prior probability and sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% same as 'sample_params_hsmm' in sampling algorithm, but use multiple
% chains to generate samples, M is the number of chains, samples are
% collected as soon as burnin is finished. Also add a llh_flag to turn on
% and off the computation of llh
%%% also add weight to the likelihood, resulting in weighted posterior

if nargin < 8
    dura_type = 'Multinomial';
end
if nargin < 9
    burnin = round(M/2);
end
if nargin < 10
    llh_flag = false;
end
if nargin < 11
    weight = ones(length(dataset),1);
end
% dataset must be an array of cells
N = length(dataset); % number of sequences in dataset
params_samples = repmat(struct(''),M,1);
states = cell(M,1);
params_samples_all = repmat(struct(''),M,burnin);
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
% LL = chol(hyperparams.emis.S);

llh = zeros(M,burnin);% 1

for m = 1:M % number of MC samples
    
    params_samples(m).params = params;
    params_samples_all(m,1).params = params;
    params_m = params;
    %~ debug
    % use sample from prior as initialization of parameters
%     for q = 1:Q
%         params_m.transmat(q,:) = dirichletrnd(hyperparams.trans(q,:));    
%         if strcmp(dura_type,'Multinomial')
%             params_m.duramat(q,1:L) = dirichletrnd(hyperparams.dura(q,1:L));
%         end
%         % emission mean
%         params_m.mu(:,q) = LL'*randn(O,1) + mu0;  
%         % params_m.sigma ??
%     end
%     if strcmp(dura_type,'Poisson')
%         params_m.duramat = gamrnd(hyperparams.dura(:,1), 1./hyperparams.dura(:,2)); % Qx1
%     end
    if llh_flag
        temp = compute_llh_evidence_HSMM(dataset, params_m, L, 'dura_type', dura_type);
        llh(m,1) = mean(temp(~isinf(temp)));
    end
    %~

    for t = 2:burnin
        %%%%%%%%%%%%% may need multiple iterations on this
        %params_samples(m).params = params_samples(m-1).params;
        % sample hidden states
        if strcmp(dura_type,'Multinomial')
            duramat_table = params_m.duramat; % QxL
        elseif strcmp(dura_type,'Poisson')
            temp = params_m.duramat(:,ones(1,L)).^const1 .* exp(-params_m.duramat(:,ones(1,L))) ./ const2;
            temp(isnan(temp)) = 0;
            temp(isinf(temp)) = 0;
            duramat_table = mk_stochastic(temp);% normalize the Poisson probability: QxL        
        else
            error('Unsupported duration distribution.')
        end
        %for re = 1:10
        [z,d] = sample_state_hsmm(dataset,params.prior,params_m.transmat,duramat_table,params_m.mu,params.sigma);
        %[z,d] = sample_state_hsmm(dataset,params.prior,params_m.transmat,duramat_table,params_m.mu,params_m.sigma);
        %end
        % compute statistics
        % collect statistics
        count_ini = zeros(Q,1);
        count_trans = zeros(Q,Q);
        count_dura = zeros(Q,L);
        count_emis = zeros(Q,O); % mean
        %count_emis2 = zeros(O,O,Q); % cov
        count_z = zeros(Q,1);    
        for n = 1:N % loop through all sequences
            % initial state
            C_ini = double(z{n}(1)==Qs);
            count_ini = count_ini + weight(n)*C_ini;
            % transition
            [~,C] = est_transmat(z{n},Q);
            count_trans = count_trans + weight(n)*(C - diag(diag(C)));
            % duration
            C = est_duramat(z{n},Q,L);
            count_dura = count_dura + weight(n)*C;        
            % emission
            for q = 1:Q
                % state count              
                count_z(q) = count_z(q) + weight(n)*sum(z{n}==q);
                vec = dataset{n}(:,z{n}==q);
                count_emis(q,:) = count_emis(q,:) + weight(n)*sum(vec,2)';
                %count_emis2(:,:,q) = count_emis2(:,:,q) + vec*vec';
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
                params_m.transmat(q,:) = dirichletrnd(cc);
            end
            if strcmp(dura_type,'Multinomial')
                dd = count_dura(q,:) + hyperparams.dura(q,1:L);
                if sum(dd) > 0
                    params_m.duramat(q,1:L) = dirichletrnd(dd);
                end
            end
            % emission mean
            if count_z(q) > 0
                Lambda = inv_sigma_prior + count_z(q)*inv_sigma(:,:,q);
                Sigma = inv(Lambda);
                Mu = Sigma*(inv_sigma(:,:,q)*count_emis(q,:)' + inv_sigma_prior*mu0); % lambda(p,:) = mu_prior %%%%%%%%%%% new thought in 2017
                params_m.mu(:,q) = chol(Sigma)'*randn(O,1) + Mu;
            end
%             % emission mean and cov
%             if count_z(q) > 0
%                 kappa = hyperparams.emis.kappa + count_z(q);
%                 nu = hyperparams.emis.nu + count_z(q);
%                 x_bar = count_emis(q,:)'/count_z(q);
%                 d1 = diag(count_emis2(:,:,q) - count_z(q)*(x_bar*x_bar'));
%                 d2 = diag(hyperparams.emis.kappa*count_z(q)/kappa*((x_bar-hyperparams.emis.mu)*(x_bar-hyperparams.emis.mu)'));
%                 if min(d1) < -1e-8
%                     disp('error 1');
%                 end
%                 if min(d2) < -1e-8
%                     disp('error 2');
%                 end
%                 Lambda = hyperparams.emis.S + count_emis2(:,:,q) - count_z(q)*(x_bar*x_bar') + ...
%                     hyperparams.emis.kappa*count_z(q)/kappa*((x_bar-hyperparams.emis.mu)*(x_bar-hyperparams.emis.mu)');
%                 Mu = hyperparams.emis.kappa/kappa*hyperparams.emis.mu + count_z(q)/kappa*x_bar;
%                 Sigma = diag(diag(iwishrnd(diag(diag(Lambda)),nu))); % use diagonal covariance for better numerical stability
%                 %Sigma = iwishrnd(Lambda,nu);
%                 params_m.sigma(:,:,q) = Sigma;
%                 params_m.mu(:,q) = chol(Sigma/kappa)'*randn(O,1) + Mu;
%             end
        end
        if strcmp(dura_type,'Poisson')
            gg = sum(count_dura,2) > 0;
            params_m.duramat(gg) = gamrnd(count_dura(gg,:)*([0:L-1]') + hyperparams.dura(gg,1), 1./(sum(count_dura(gg,:),2) + hyperparams.dura(gg,2))); % Qx1
        end
        %end
        % optional: evaluate likelihood of the training data on sampled parameters
        if llh_flag
            temp = compute_llh_evidence_HSMM(dataset, params_m, L, 'dura_type', dura_type);
            llh(m,t) = mean(temp(~isinf(temp)));
        end
        params_samples_all(m,t).params = params_m;
        %
    end
    states{m} = z;
    params_samples(m).params = params_m;
    fprintf('Sample %d collected\n',m);
end