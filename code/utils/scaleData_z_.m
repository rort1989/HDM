function [featScale_z, matScale] = scaleData_z_(feat_z, meanScale, stdScale, matScale)
% scale the data in each dimension according to matScale if provided
% use z-score type of normalization
% take an array of cells as input

featScale_z = cell(size(feat_z));
feat = [];
for n = 1:length(feat_z)
    feat = [feat feat_z{n}];
end

% initialize
[ndims, nfeats] = size(feat); % dimension of feature x number of samples

%if matScale is not specified, build it from feat
if nargin < 4
    % initialize matScale
    matScale = zeros(ndims, 2);
    
    % mean value at each dimension of feat
    matScale(:, 1) = mean(feat, 2);
    
    % std value at each dimension of feat
    matScale(:, 2) = std(feat, [], 2);
end

%~ alternative implementation
mean_val = matScale(:,1);
std_val = matScale(:,2);
idx = std_val==0;

for n = 1:length(feat_z)
    temp = bsxfun(@minus, feat_z{n}, mean_val);
    temp = bsxfun(@rdivide, temp, std_val);
    featScale_z{n} = stdScale * temp + meanScale;
    if sum(idx) > 0
        featScale_z{n}(:, idx) = 0;
    end
end

end