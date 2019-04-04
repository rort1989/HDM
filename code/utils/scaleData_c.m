function [featScale_c, matScale] = scaleData_c(feat_c, minScale, maxScale, matScale)
% scale the data in each dimension according to matScale if provided
% take an array of cells as input

featScale_c = cell(size(feat_c));
feat = [];
for n = 1:length(feat_c)
    feat = [feat feat_c{n}];
end

% initialize
[ndims, nfeats] = size(feat); % dimension of feature x number of samples

%if matScale is not specified, build it from feat
if nargin < 4
    % initialize matScale
    matScale = zeros(ndims, 2);
    
    % min value at each dimension of feat
    matScale(:, 1) = min(feat, [], 2);
    
    % max value at each dimension of feat
    matScale(:, 2) = max(feat, [], 2);
end

%~ alternative implementation
minval = matScale(:,1);
maxval = matScale(:,2);
scale = maxval - minval;
idx = scale==0;

for n = 1:length(feat_c)
    temp = bsxfun(@minus, feat_c{n}, minval);
    temp = bsxfun(@rdivide, temp, scale);
    featScale_c{n} = (maxScale - minScale) * temp + minScale;
    if sum(idx) > 0
        featScale_c{n}(idx,:) = 0;
    end
end

end