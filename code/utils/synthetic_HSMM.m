function datasets = synthetic_HSMM(num_class,numex,T,num_hstate,dim_observed,varargin)
% generate sythentic datasets using HSMM. 
% datasets = synthetic_HMM(num_class,numex,T,num_hstate,dim_observed,varargin)
% Optional input:
%         'discrete', 'obsmat', 'mixmat', 'scale_mu', 'scale_sigma',
%         'kappa', 'params', 'variant'
% Support discrete observation node with multi-nomial distribution 
% or continuous observation node with mixture of Gaussian distribution
% the 'variant' specifies the duration type of model: Gamma, Categorical,
% Poisson

datasets = repmat(struct('observed',[],'hidden',[],'prior',[],'transmat',[],'duramat',[],'obsmat',[],'mu',[],'sigma',[],'mixmat',[]),1,num_class);
Q = num_hstate; % num hidden states
O = dim_observed; %  num observable symbols
p = inputParser;
default_discrete = 0;
default_scale_mu = 10;
default_scale_sigma = 9;
default_kappa = 0;
default_params = [];
default_variant = 1;
addOptional(p,'discrete',default_discrete,@isnumeric);
addOptional(p,'scale_mu',default_scale_mu,@isnumeric);
addOptional(p,'scale_sigma',default_scale_sigma,@isnumeric);
addOptional(p,'kappa',default_kappa,@isnumeric);
addOptional(p,'params',default_params,@isstruct);
addOptional(p,'variant',default_variant,@isnumeric);
p.parse(varargin{:});
discrete = p.Results.discrete;
scale_mu = p.Results.scale_mu;
scale_sigma = p.Results.scale_sigma;
kappa = p.Results.kappa;
params = p.Results.params;
variant = p.Results.variant;

for i = 1:num_class 
    if ~isempty(params) % isstruct(params(i))
        prior = params(i).prior;
        transmat = params(i).transmat;
        duramat = params(i).duramat;
    else
        prior = normalise(rand(Q,1));
        transmat = mk_stochastic(rand(Q,Q)+kappa*eye(Q)); % +(i-1)*eye(Q) % add more self transition weight with larger class idx i
        if variant == 1 % gamma
            duramat = [5*[1:Q]' ones(Q,1)];
        elseif variant == 2 % categorical
            duramat = mk_stochastic(repmat(T:-1:1,Q,1).*repmat([1:Q]',1,T));
        else % poisson
            duramat = 3*[1:Q]';
        end
        % avoid self-transition
        transmat = transmat - diag(diag(transmat));
        transmat = mk_stochastic(transmat);
        % in case the original trans only has self-transition
        transmat = transmat - diag(diag(transmat));
        transmat = mk_stochastic(transmat);
    end
    if ~discrete
        if ~isempty(params) % isstruct(params(i))
            mu = params(i).mu;
            sigma = params(i).sigma;
            if isfield(params(i),'mixmat')
                mixmat = params(i).mixmat;
                M = size(params(i).mixmat,2);
            else
                M = 1;
                mixmat = ones(Q,1);
            end
        else
            M = 1;
            mu = repmat(scale_mu*(1:Q),O,1); %.*rand(O,Q)
            mu = repmat(mu,[1 1 M]);
            sigma = repmat(scale_sigma*eye(O), [1,1,Q,M]); % rand
            mixmat = ones(Q,1);
        end
        [obs, hidden] = mhsmm_sample(T, numex, prior, transmat, duramat, mu, sigma, mixmat, variant);
    else
        if isstruct(params(i))
            obsmat = params(i).obsmat;
        else
            obsmat = mk_stochastic(rand(Q,O));
        end
        [obs, hidden] = dhsmm_sample(prior, transmat, duramat, obsmat, numex, T, variant);
    end
    datasets(i).observed = obs;    
    datasets(i).hidden = hidden;
    datasets(i).prior = prior;
    datasets(i).transmat = transmat;
    datasets(i).duramat = duramat;
    if discrete        
        datasets(i).obsmat = obsmat;
    else
        datasets(i).mu = mu;
        datasets(i).sigma = sigma;
        datasets(i).mixmat = mixmat;
    end
end

end
