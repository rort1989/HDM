function [mu, sigma] = hsmm_mstep_emis(N_ess, m_ess, V_ess, cov_type, hyperparams)
% perform M-step in EM algorithm given expected sufficient statistics on
% Gaussian emission distribution parameters and optionally hyperparameters
% to get new parameter values 

if nargin < 5
    flag_hyper = 0;
else
    flag_hyper = 1;
end

[O,Q] = size(m_ess);

if flag_hyper == 0
    hyperparams.emis.kappa = 0;
    hyperparams.emis.mu = 0;
    hyperparams.emis.nu = -2-O;
    hyperparams.emis.S = zeros(O);    
end

kappa = hyperparams.emis.kappa;
mu0 = hyperparams.emis.mu;
nu0 = hyperparams.emis.nu;
S = hyperparams.emis.S;
mu = zeros(O,Q);
sigma = zeros(O,O,Q);
% complete after revision of submission (see supplementary for CVPR2017)
NT = N_ess+kappa;
NN = N_ess+nu0+O+2;
if size(N_ess,2) == 1 % complete data case
    for q = 1:Q
        kmu0 = kappa*mu0;
        MT = m_ess(:,q) + kmu0;
        mu(:,q) = MT / NT(q);
        sigma(:,:,q) = V_ess(:,:,q) / NN(q) - mu(:,q)*MT' / NN(q) + (kmu0*mu0' + S) / NN(q);
        % force covariance matrix to be diagonal if set (bnt: 'mixgauss_Mstep')
        if strcmp(cov_type,'diag')
            SS = sigma(:,:,q);
            sigma(:,:,q) = diag(diag(SS));
        end
    end
elseif size(N_ess,2) == Q % missing data case
    for q = 1:Q
        kmu0 = kappa*mu0;
        MT = m_ess(:,q) + kmu0;
        mu(:,q) = MT ./ NT(:,q);
        SS = diag(V_ess(:,:,q))./NN(:,q) - mu(:,q).*MT./NN(:,q) + (kmu0.*mu0+diag(S))./NN(:,q);
        % force covariance matrix to be diagonal if contains missing data       
        sigma(:,:,q) = diag(SS);
    end
end