function [featScale, meanData, varData, varProj, w, eigvalue, energy, idx_end, idx_nonconst] = scaleData_PCAw(feat_c, portion, meanData, w, eigvalue, idx_nonconst)
% compute PCA projection of data + whitening on projected data (set the covariance matrix to be identity)

if nargin < 3
    performPCA = 1;
else 
    performPCA = 0;
end

if performPCA == 1 
    % compute PCA projection coefficient
    feat = zeros(size(feat_c{1},1),10000);
    idx_end = 0;
    for n = 1:length(feat_c)
        T = size(feat_c{n},2);
        feat(:,1+idx_end:T+idx_end) = feat_c{n};
        idx_end = idx_end + T;
    end
    feat = feat(:,1:idx_end);    
    stdData = std(feat,[],2);
    idx_nonconst = stdData>1e-8;
    D_low = sum(idx_nonconst);
    [w, eigvalue, proj, meanData, varData] = myPCA(feat(idx_nonconst,:), D_low, 0);    
    clear feat
    varProj = var(proj,[],2);
    energy = cumsum(eigvalue)./sum(eigvalue);
    dim = find(energy>portion,1);
    w = w(:,1:dim);
    eigvalue = eigvalue(1:dim);
end
% compute linear projection
featScale = cell(size(feat_c));
for n = 1:length(feat_c)
    featScale{n} = diag(eigvalue.^(-0.5))*w'*bsxfun(@minus,feat_c{n}(idx_nonconst,:),meanData);
end