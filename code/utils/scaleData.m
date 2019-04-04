function [featScale, matScale] = scaleData(feat, minScale, maxScale, matScale)
% original author: Xiaodong Yang

% initialize
[nfeats, ndims] = size(feat); % number of samples x dimension of feature
%~ featScale = zeros(nfeats,ndims);

%if matScale is not specified, build it from feat
if nargin < 4
    % initialize matScale
    matScale = zeros(2,ndims);
    
    % min value at each dimension of feat
    matScale(1, :) = min(feat, [], 1);
    
    % max value at each dimension of feat
    matScale(2, :) = max(feat, [], 1);
end

% % each dimension to the same scale and offset.
% for i = 1:ndims
%     col = feat(:, i);
%     minval = matScale(1, i);
%     maxval = matScale(2, i);
%     
%     if (maxval - minval) ~= 0
%         colScale = (maxScale - minScale) / (maxval - minval) * (col - minval) + minScale;
%     else
%         colScale = zeros(length(col), 1);
%     end
%     
%     featScale(:, i) = colScale;
% end

%~ alternative implementation
minval = matScale(1, :);
maxval = matScale(2, :);
scale = maxval - minval;
idx = scale==0;
temp = bsxfun(@minus, feat, minval);
temp = bsxfun(@rdivide, temp, scale);
featScale = (maxScale - minScale) * temp + minScale;
if sum(idx) > 0
    featScale(:, idx) = 0;
end

end