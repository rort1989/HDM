function featScale_z = scaleData_z_r(feat_z, matScale)
% re-scale the data in each dimension according to matScale
% use z-score type of normalization
% take an array of cells as input

featScale_z = cell(size(feat_z));
meanScale = matScale(:,1);
stdScale = matScale(:,2);

for n = 1:length(feat_z)
    temp = bsxfun(@times, feat_z{n}, stdScale);
    featScale_z{n} = bsxfun(@plus, temp, meanScale);
end

end