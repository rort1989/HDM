function labels = sc(data, K, varargin)
% perform spectral clustering on data using NJW [1] algorithm
% input:
%       data - N*M matrix where N is number of instances, M is dimension of
%                 datapoints
%       K - number of clusters
%       option -  
%           'normalization': normalization for each dimension of feature
%           within data (default: 0)
%           'sigma': parameter values (default: mean vector 2-norm of data)
%           'rep': kmeans replications (default: 10)
% output:
%       labels - N*1 vectors where each entry has discrete value between 1
%       to K
%
% reference:
% [1] A. Ng, M. Jordan, Y. Weiss, On spectral clustering: Analysis and an
% algorithm, NIPS 2002

% load optional arguments
sigma = mean(sqrt(sum(data.^2,2)));
rep = 10;
normalization = 0;
for argidx = 1:2:nargin-2
    switch varargin{argidx}
        case 'sigma'
            sigma = varargin{argidx+1};
        case 'rep'
            rep = varargin{argidx+1};
        case 'normalization'
            normalization = varargin{argidx+1};
    end
end

% Step 1. form affinity matrix
[N,M] = size(data);
if normalization % scale between 0 to 1
    temp = bsxfun(@minus,data,min(data,2));
    data = bsxfun(@rdivide,temp,max(data,2)-min(data,2));
end
S = data*data';
D = diag(S);
A = -2*S; % size N*N
A = bsxfun(@plus,A,D(:));
A = bsxfun(@plus,A,D(:)'); % by now A has zero along main diagonal
A = exp(-A/2/sigma^2); % take exponential for off-diagonal entries
A = A - eye(N); % zero out diagonal entries

% Step 2. form L matrix
D = sum(A,2);
L = diag(D.^(-0.5))*A*diag(D.^(-0.5));

% Step 3. find the eigenvectors corresponding to K largest eigenvalues
[V,E] = eigs(L,K); %~ the largest eigenvalue must be 1
nfactor = sqrt(sum(V.^2,2));
V = bsxfun(@rdivide,V,nfactor); 

% Step 4. K-means clustering on V's rows
labels = kmeans(V,K,'Replicates',rep,'EmptyAction','drop');

end