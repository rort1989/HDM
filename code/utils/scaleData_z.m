function [featScale_z, matScale] = scaleData_z(feat_z, meanScale, stdScale, matScale)
% use z-score type of normalization
% take an array of cells as input

% initialize
featScale_z = cell(size(feat_z));
ndims = size(feat_z{1},1);

%if matScale is not specified, build it from feat
if nargin < 4
    % initialize matScale
    matScale = zeros(ndims, 2);
    feat_z_all = zeros(ndims,10000);
    T_count = 0;
    for n = 1:length(feat_z)
        T = size(feat_z{n},2);
        feat_z_all(:,T_count+1:T_count+T) = feat_z{n};
        T_count = T_count + T;
    end
    feat_z_all = feat_z_all(:,1:T_count);
    
    % get all the data
    % mean value at each dimension of feat
    matScale(:,1) = mean(feat_z_all,2);
    
    % std value at each dimension of feat
    matScale(:,2) = std(feat_z_all, [], 2);
end

% scale each data.
idx_valid = matScale(:,2) > 0; % can change to a larger number
for n = 1:length(featScale_z)
    
    temp = bsxfun(@minus, feat_z{n}, matScale(:,1));  
    temp(idx_valid,:) = bsxfun(@rdivide, temp(idx_valid,:), matScale(idx_valid,2));
    featScale_z{n} = stdScale * temp + meanScale;
    
end

end