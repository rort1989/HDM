function [llhh,llhh_prior,llhh_trans,llhh_dura,llhh_emis_mu,llhh_emis_sigma] = llh_hyperparams_hsmm(params,hyperparams,dura_type)

% hyperparameters
eta0 = hyperparams.init; % zeros(Q,1);
eta = hyperparams.trans; % zeros(Q,Q); % subtraction for imposing no prior on self-transition
xi = hyperparams.dura; % zeros(Q,L); or zeros(Q,2) % need this prior to handle non-existing
kappa0 = hyperparams.emis.kappa;
mu0 = hyperparams.emis.mu;
Sigma0 = hyperparams.emis.S;
nu0 = hyperparams.emis.nu;
% parameters
prior = params.prior;
transmat = params.transmat;
duramat = params.duramat; % depending on dura_type, this could be QxL (Multinomial) or Qx1 (Poisson), specifying the dura distribution parameters
mu = params.mu;
sigma = params.sigma;
Q = length(prior);
llhh = 0;

% % prior: Dirichlet
llhh_prior = 0;
% llhh_prior = logDirichlet(prior, eta0);
% llhh = llhh_prior;
% transmat: Dirichlet
llhh_trans = 0;
for q = 1:Q
    llhh_trans = llhh_trans + logDirichlet(transmat(q,[1:q-1 q+1:Q])', eta(q,[1:q-1 q+1:Q])');
end
llhh = llhh + llhh_trans;
% duration: Gamma
if strcmp(dura_type,'Multinomial')
    temp = logDirichlet(duramat', xi');
    llhh_dura = trace(temp);
elseif strcmp(dura_type,'Poisson')
    temp = logGamma(duramat, xi(:,1), xi(:,2));
    llhh_dura = sum(temp);
end
llhh = llhh + llhh_dura;
% emission: NIW
temp = logNormal(mu, mu0, Sigma0/kappa0);
llhh_emis_mu = sum(temp);
llhh = llhh + llhh_emis_mu;
llhh_emis_sigma = 0;
% for q = 1:Q    
%     llhh_emis_sigma = llhh_emis_sigma + logIWishart(sigma(:,:,q), Sigma0, nu0);
% end
% llhh = llhh + llhh_emis_sigma;