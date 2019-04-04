function [obslik, frame_not_missing] = mixgauss_prob_miss(obs, mu, sigma, mask_missing)

[~,T] = size(obs);
[O,Q] = size(mu);
obslik = zeros(Q,T); % P(X_t|Z_t=q)
frame_not_missing = ones(1,T,'logical');
if isempty(mask_missing) % complete data
    idx_complete = true(1,size(obs,2));
else
    idx_complete = sum(mask_missing) == O; % mask_missing is binary matrix of OxT
end
idx_incomplete = find(idx_complete==0);
if sum(idx_complete) > 0
    obslik(:,idx_complete) = mixgauss_prob(obs(:,idx_complete), mu, sigma);
end
for i = 1:length(idx_incomplete) %%%%%% assert if all data are observed, the following loop will not be executed
    idx_obs = boolean(mask_missing(:,idx_incomplete(i)));
    if sum(idx_obs) == 0 % whole frame is missing
        frame_not_missing(idx_incomplete(i)) = false;
        continue;
    end
    obslik(:,idx_incomplete(i)) = mixgauss_prob(obs(idx_obs,idx_incomplete(i)), mu(idx_obs,:), sigma(idx_obs,idx_obs,:));
end
obslik = obslik(:,frame_not_missing);