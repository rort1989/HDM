function [prior, transmat, duramat] = hsmm_mstep_dynm(ini_ess, tr_ess, dr_ess, L, hyperparams, dura_type)
% perform M-step in EM algorithm given expected sufficient statistics on
% initial, transition and duration parameters and optionally
% hyperparameters to get new parameter values 

if nargin < 5
    flag_hyper = 0;
else
    flag_hyper = 1;
end
if nargin < 6
    dura_type = 'Multinomial';
end

[Q,T] = size(dr_ess);

if flag_hyper == 0
    hyperparams.init = ones(Q,1);
    hyperparams.trans = ones(Q,Q); 
    % there are two cases: 1. insymmetric prior, where individual entry
    % value can be different, 2. symmetric prior, where entry of each row
    % is the same. However in both case, the diagonal entry remain 0
    hyperparams.dura = ones(Q,L);
end

transmat = zeros(Q,Q);
%%%%%%%%%%%%%% use MAP estimate for update
% prior = (ini_ess + hyperparams.init - 1) / (sum(ini_ess) + sum(hyperparams.init) - Q);
% for q = 1:Q
%     transmat(q,:) = (tr_ess(q,:) + hyperparams.trans(q,:) - 1) / (sum(tr_ess(q,:)) + sum(hyperparams.trans(q,:)) - Q);
%     duramat(q,:) = (dr_ess(q,:) + hyperparams.dura(q,:) - 1) / (sum(dr_ess(q,:)) + sum(hyperparams.dura(q,:)) - L);   
% end
%%%%%%%%%%%%%% use posterior mean for update (when data is sparse)
prior = (ini_ess + hyperparams.init) / (sum(ini_ess) + sum(hyperparams.init) );
if strcmp(dura_type,'Multinomial')
    duramat = zeros(Q,L);
    for q = 1:Q
        transmat(q,:) = (tr_ess(q,:) + hyperparams.trans(q,:)) / (sum(tr_ess(q,:)) + sum(hyperparams.trans(q,:)));
        duramat(q,1:T) = (dr_ess(q,:) + hyperparams.dura(q,1:T)) / (sum(dr_ess(q,:)) + sum(hyperparams.dura(q,1:T)));   
    end
else % Poisson  also use posterior mean for update
    duramat = zeros(Q,1);
    for q = 1:Q
        transmat(q,:) = (tr_ess(q,:) + hyperparams.trans(q,:)) / (sum(tr_ess(q,:)) + sum(hyperparams.trans(q,:)));
        duramat(q) = (dr_ess(q,:) * [0:T-1]' + hyperparams.dura(q,1)) / (sum(dr_ess(q,:)) + hyperparams.dura(q,2)); %%%%%% assert this is a positive number
    end
end

    