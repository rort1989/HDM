function datasets = synthetic_HMM(num_class,numex,T,num_hstate,dim_observed,varargin)
% generate sythentic datasets using HMM. 
% datasets = synthetic_HMM(num_class,numex,T,num_hstate,dim_observed,varargin)
% Optional input:
%         'discrete', 'obsmat', 'mixmat', 'scale_mu', 'scale_sigma',
%         'kappa', 'params'
% Support discrete observation node with multi-nomial distribution 
% or continuous observation node with mixture of Gaussian distribution

datasets = repmat(struct('observed',[],'hidden',[],'prior',[],'transmat',[],'obsmat',[],'mu',[],'sigma',[],'mixmat',[]),1,num_class);
Q = num_hstate; % num hidden states
O = dim_observed; %  num observable symbols
p = inputParser;
default_discrete = 0;
default_scale_mu = 10;
default_scale_sigma = 9;
default_kappa = 0;
default_params = [];
addOptional(p,'discrete',default_discrete,@isnumeric);
addOptional(p,'scale_mu',default_scale_mu,@isnumeric);
addOptional(p,'scale_sigma',default_scale_sigma,@isnumeric);
addOptional(p,'kappa',default_kappa,@isnumeric);
addOptional(p,'params',default_params,@isstruct);
p.parse(varargin{:});
discrete = p.Results.discrete;
scale_mu = p.Results.scale_mu;
scale_sigma = p.Results.scale_sigma;
kappa = p.Results.kappa;
params = p.Results.params;

for i = 1:num_class
    if ~isempty(params) % isstruct(params(i))
        prior = params(i).prior;
        transmat = params(i).transmat;        
    else
        prior = normalise(rand(Q,1));
        transmat = mk_stochastic(rand(Q,Q)+kappa*eye(Q)); % +(i-1)*eye(Q) % add more self transition weight with larger class idx i
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
            mu = zeros(O,Q,M);
            for m = 1:M
                mu(:,:,m) = repmat(m+scale_mu*(1:Q),O,1); %.*rand(O,Q)
            end
            sigma = repmat(scale_sigma*eye(O), [1,1,Q,M]); % rand            
            mixmat = ones(Q,1);
        end
        [obs, hidden] = mhmm_sample(T, numex, prior, transmat, mu, sigma, mixmat);
    else
        if ~isempty(params) % isstruct(params(i))
            obsmat = params(i).obsmat;
        else
            obsmat = mk_stochastic(rand(Q,O));
        end
        [obs, hidden] = dhmm_sample(prior, transmat, obsmat, numex, T);
    end
    datasets(i).observed = obs;
    datasets(i).prior = prior;
    datasets(i).transmat = transmat;
    if discrete
        datasets(i).hidden = hidden;
        datasets(i).obsmat = obsmat;
    else
        datasets(i).hidden = hidden';
        datasets(i).mu = mu;
        datasets(i).sigma = sigma;
        datasets(i).mixmat = mixmat;
    end
end

end
