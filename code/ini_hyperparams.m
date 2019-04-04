function hyperparams = ini_hyperparams(Q,O,L,dura_prior_coeff,useprior,cov_prior,dura_type)

if nargin < 7
    dura_type = 'Multinomial';
end

hyperparams.init = 0.01*ones(Q,1);
hyperparams.trans = 0.01*ones(Q,Q) - 0.01*eye(Q); % subtraction for imposing no prior on self-transition
if strcmp(dura_type,'Multinomial')
    hyperparams.dura = dura_prior_coeff*ones(Q,L); % dura_prior_coeff need this prior to handle non-existing
elseif strcmp(dura_type,'Poisson')
    hyperparams.dura = dura_prior_coeff*ones(Q,2);
else
    error('Unsupported duration distribution.')
end
hyperparams.emis.kappa = 0;
hyperparams.emis.mu = 0;
hyperparams.emis.S = cov_prior;
hyperparams.emis.nu = 2+O;

if useprior == 0
    hyperparams.init = zeros(Q,1);
    hyperparams.trans = zeros(Q,Q); % subtraction for imposing no prior on self-transition
    hyperparams.dura = zeros(Q,L); % need this prior to handle non-existing
    hyperparams.emis.kappa = 0;
    hyperparams.emis.mu = 0;
    hyperparams.emis.S = 0;
    hyperparams.emis.nu = -2-O;
elseif useprior == 4 % only use spatial prior
    hyperparams.init = zeros(Q,1);
    hyperparams.trans = zeros(Q,Q); % subtraction for imposing no prior on self-transition
    hyperparams.dura = zeros(Q,L); % need this prior to handle non-existing
elseif useprior == 3 % only use temporal prior
    hyperparams.emis.kappa = 0;
    hyperparams.emis.mu = 0;
    hyperparams.emis.S = 0;
    hyperparams.emis.nu = -2-O;
elseif useprior == 5 % only use duration prior
    hyperparams.init = zeros(Q,1);
    hyperparams.trans = zeros(Q,Q); % subtraction for imposing no prior on self-transition
    hyperparams.emis.kappa = 0;
    hyperparams.emis.mu = 0;
    hyperparams.emis.S = 0;
    hyperparams.emis.nu = -2-O;
end