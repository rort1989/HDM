% rng default;
feature = cell(num_inst,1);
nframe_count = zeros(num_inst,1); % original nframe_count set to zero as we may downsample sequence to get new nframe_count
% mask_missing = cell(1);
if config == 0
    if isempty(d_selected)
        d_selected = 1:size(feature1_{1},1);        
    end
    O_ = length(d_selected);
    % if applied, randomly set observations to 0 as missing values:
    % this process only applies to joint position feature and the
    % corresponding motion feature will be set to 0 as well
%     mask_missing = cell(num_inst,1); 
    if exist('feature2_','var')
        for n = 1:num_inst
            T = min(size(feature1_{n},2),size(feature2_{n},2));
            T_ = length(1:stride:T);
%             mask_missing{n} = double(rand(O_,T_) >= portion_missing);
%             mask_missing{n} = [mask_missing{n}; mask_missing{n}];
            feature{n} = [feature1_{n}(d_selected,1:stride:T); feature2_{n}(diag(lut_motions),1:stride:T)]; %d_selected .* mask_missing{n}
            nframe_count(n) = size(feature{n},2);
        end
    else
        for n = 1:num_inst
            T = size(feature1_{n},2);
            T_ = length(1:stride:T);
%             mask_missing{n} = double(rand(O_,T_) >= portion_missing);
            feature{n} = feature1_{n}(d_selected,1:stride:end); % .* mask_missing{n}
            nframe_count(n) = size(feature{n},2);
        end
    end
    %cov_prior_coeff = 0.01;% defined in configuration file
elseif config == 1
    feature1 = feature;
    feature2 = feature;
    [feature1(idx_train), meanData1, varData1, varProj1, w1, energy1, idx_end1, idx_nonconst1] = scaleData_PCA(feature1_(idx_train), portion);
    feature1(idx_test) = scaleData_PCA(feature1_(idx_test), portion, meanData1, w1, idx_nonconst1);
    [feature2(idx_train), meanData2, varData2, varProj2, w2, energy2, idx_end2, idx_nonconst2] = scaleData_PCA(feature2_(idx_train), portion);
    feature2(idx_test) = scaleData_PCA(feature2_(idx_test), portion, meanData2, w2, idx_nonconst2);
    O_ = size(feature1{idx_train(1)},1) + size(feature2{idx_train(1)},1);
%     mask_missing = cell(num_inst,1); 
    for n = 1:num_inst
        T = min(size(feature1{n},2),size(feature2{n},2));
        T_ = length(1:stride:T);
%         mask_missing{n} = double(rand(O_,T_) >= portion_missing);
        feature{n} = [feature1{n}(:,1:stride:T); feature2{n}(:,1:stride:T)];% .* mask_missing{n};
        nframe_count(n) = T;
    end
    O1 = size(feature1{idx_train(1)},1);
    O2 = size(feature2{idx_train(1)},1);
    cov_prior_coeff = cov_prior_coeff_portion*diag([sqrt(varProj1(1:O1)); sqrt(varProj2(1:O2))]);
    clear feature1 feature2
elseif config == 2
    feature1 = feature;
    feature2 = feature;
    [feature1(idx_train), meanData1, varData1, varProj1, w1, eigvalue1, energy1, idx_end1, idx_nonconst1] = scaleData_PCAw(feature1_(idx_train), portion);
    feature1(idx_test) = scaleData_PCAw(feature1_(idx_test), portion, meanData1, w1, eigvalue1, idx_nonconst1);
    [feature2(idx_train), meanData2, varData2, varProj2, w2, eigvalue2, energy2, idx_end2, idx_nonconst2] = scaleData_PCAw(feature2_(idx_train), portion);
    feature2(idx_test) = scaleData_PCAw(feature2_(idx_test), portion, meanData2, w2, eigvalue2, idx_nonconst2);
    for n = 1:num_inst
        T = min(size(feature1{n},2),size(feature2{n},2));
        feature{n} = [feature1{n}(:,1:stride:T); feature2{n}(:,1:stride:T)];
        nframe_count(n) = T;
    end
    cov_prior_coeff = 0.3;
    clear feature1 feature2
elseif config == 3
    feature1 = feature;    
    [feature1(idx_train), meanData1, varData1, varProj1, w1, energy1, idx_end1, idx_nonconst1] = scaleData_PCA(feature1_(idx_train), portion);
    feature1(idx_test) = scaleData_PCA(feature1_(idx_test), portion, meanData1, w1, idx_nonconst1);    
    for n = 1:num_inst
        feature{n} = feature1{n}(:,1:stride:end);
        nframe_count(n) = size(feature{n},2);
    end
    O1 = size(feature1{idx_train(1)},1);
    cov_prior_coeff = cov_prior_coeff_portion*diag(sqrt(varProj1(1:O1)));
    clear feature1
elseif config == 4
    feature1 = feature;    
    [feature1(idx_train), meanData1, varData1, varProj1, w1, eigvalue1, energy1, idx_end1, idx_nonconst1] = scaleData_PCAw(feature1_(idx_train), portion);
    feature1(idx_test) = scaleData_PCAw(feature1_(idx_test), portion, meanData1, w1, eigvalue1, idx_nonconst1); 
    for n = 1:num_inst
        feature{n} = feature1{n}(:,1:stride:end);
        nframe_count(n) = size(feature{n},2);
    end
    cov_prior_coeff = 0.3;
    clear feature1
end