function [params, map_hstate] = estParams(params_set,loglikelihood,hstate,discrete,method)
% estimate HMM parameters from sampling results
% Issues considered: discrete vs. continuous, finite vs. HDP, order
% the hidden state assignment is exchangeable
% sort the paramters according to hidden state initial probability

num_trial = length(params_set);
num_iter = length(params_set(1).params);
% determine the number of hidden state as the majority cardinality
% of hidden state among different trials and iterations
num_hstate = zeros(1,hstate);
map_hstate = zeros(num_iter, num_trial);
for t = 1:num_trial
    for i = 1:num_iter
        j = length(params_set(t).params(i).prior);
        map_hstate(i,t) = j;
        if j > hstate, error('Number of state exceed upper bound'); end
        num_hstate(j) = num_hstate(j) + 1;
    end
end
%~ check point, sum(sum(num_hstate)) = num_iter*num_trial
[~,Q] = max(num_hstate);
%~ manually decide the hstate number
% Q = 3;
if discrete
    O = size(params_set(1).params(1).obsmat,2);
else
    O = size(params_set(1).params(1).mu,1);
    M = size(params_set(1).params(1).mixmat,2);
end
% define parameter struct
params.prior = zeros(Q,1);
params.transmat = zeros(Q,Q);
if discrete
    params.obsmat = zeros(Q,O);
else
    params.mu = zeros(O,Q,M);
    params.sigma = zeros(O,O,Q,M);
    params.mixmat = zeros(Q,M);
end
params.count = 0;

if method == 1 % use average of last iteration of all the trials with hidden state equals to the majority    
    % average over those trials whose hidden state equals to the num_hstate    
    for t = 1:num_trial
        if length(params_set(t).params(num_iter).prior) == Q % only use selected trials
            % sorted according to the same assignment scheme
            [~,order] = sort(params_set(t).params(num_iter).mu(1,:)); %~ this sorting can be problematic
            params.prior = params.prior + params_set(t).params(num_iter).prior(order);
            params.transmat = params.transmat + params_set(t).params(num_iter).transmat(order,order);
            if discrete
                params.obsmat = params.obsmat + params_set(t).params(num_iter).obsmat(order,:);
            else
                params.mu = params.mu + params_set(t).params(num_iter).mu(:,order);
                params.sigma = params.sigma + params_set(t).params(num_iter).sigma(:,:,order,:);
                params.mixmat = params.mixmat + params_set(t).params(num_iter).mixmat(order,:);
            end
            params.count = params.count + 1;
        end
    end
    params.prior = params.prior/params.count;
    params.transmat = params.transmat/params.count;
    if discrete
        params.obsmat = params.obsmat/params.count;
    else
        params.mu = params.mu/params.count;
        params.sigma = params.sigma/params.count;
        params.mixmat = params.mixmat/params.count;
    end

elseif method == 2 % use average of the trial with largest average llh
    idx_selected = map_hstate==Q; % qualified iteration
    llh_selected = loglikelihood.*double(idx_selected);    % only use iterations with designated number of hidden state
    llh_avg = sum(llh_selected)./sum(idx_selected); % 1*num_trial vector
    [~,opt_trial] = max(llh_avg);
    for i = 1:num_iter
        if idx_selected(i,opt_trial) % only use qualified iteration in optimal trial
            % sorted according to the same assignment scheme
            if discrete
                [~,order] = sort(params_set(opt_trial).params(i).prior);
            else
                [~,order] = sort(params_set(opt_trial).params(i).mu(1,:)); %~ this sorting can be problematic
            end
            params.prior = params.prior + params_set(opt_trial).params(i).prior(order);
            params.transmat = params.transmat + params_set(opt_trial).params(i).transmat(order,order);
            if discrete
                params.obsmat = params.obsmat + params_set(opt_trial).params(i).obsmat(order,:);
            else
                params.mu = params.mu + params_set(opt_trial).params(i).mu(:,order);
                params.sigma = params.sigma + params_set(opt_trial).params(i).sigma(:,:,order,:);
                params.mixmat = params.mixmat + params_set(opt_trial).params(i).mixmat(order,:);
            end
            params.count = params.count + 1;
        end
    end
    params.prior = params.prior/params.count;
    params.transmat = params.transmat/params.count;
    if discrete
        params.obsmat = params.obsmat/params.count;
    else
        params.mu = params.mu/params.count;
        params.sigma = params.sigma/params.count;
        params.mixmat = params.mixmat/params.count;
    end
    
elseif method == 3 % similar to method 2 but using weighted average
    idx_selected = map_hstate==Q; % qualified iteration
    llh_selected = loglikelihood.*double(idx_selected);    % only use iterations with designated number of hidden state
    llh_avg = sum(llh_selected)./sum(idx_selected); % 1*num_trial vector
    [~,opt_trial] = max(llh_avg);
    max_llh = max(loglikelihood(idx_selected(:,opt_trial),opt_trial));
    llh_offset = loglikelihood(idx_selected(:,opt_trial),opt_trial) - max_llh;
    p_offset = exp(llh_offset);
    wl = p_offset/sum(p_offset);
    for i = 1:num_iter
        if idx_selected(i,opt_trial) % only use qualified iteration in optimal trial
            params.count = params.count + 1;
            if discrete
                [~,order] = sort(params_set(opt_trial).params(i).prior);
            else
                [~,order] = sort(params_set(opt_trial).params(i).mu(1,:)); %~ this sorting can be problematic
            end
            params.prior = params.prior + params_set(opt_trial).params(i).prior(order)*wl(params.count);
            params.transmat = params.transmat + params_set(opt_trial).params(i).transmat(order,order)*wl(params.count);
            if discrete
                params.obsmat = params.obsmat + params_set(opt_trial).params(i).obsmat(order,:)*wl(params.count);
            else
                params.mu = params.mu + params_set(opt_trial).params(i).mu(:,order)*wl(params.count);
                params.sigma = params.sigma + params_set(opt_trial).params(i).sigma(:,:,order,:)*wl(params.count);
                params.mixmat = params.mixmat + params_set(opt_trial).params(i).mixmat(order,:)*wl(params.count);
            end            
        end
    end    
    
elseif method == 4 % use average of different trials and different iterations
    idx_selected = map_hstate==Q; % qualified iteration
    for t = 1:num_trial
        for i = 1:num_iter
            if idx_selected(i,t) % only use qualified iteration in each trial
                % sorted according to the same assignment scheme
                [~,order] = sort(params_set(t).params(i).prior); %~ mu(1,:) this sorting can be problematic
                params.prior = params.prior + params_set(t).params(i).prior(order);
                params.transmat = params.transmat + params_set(t).params(i).transmat(order,order);
                if discrete
                    params.obsmat = params.obsmat + params_set(t).params(i).obsmat(order,:);
                else
                    params.mu = params.mu + params_set(t).params(i).mu(:,order);
                    params.sigma = params.sigma + params_set(t).params(i).sigma(:,:,order,:);
                    params.mixmat = params.mixmat + params_set(t).params(i).mixmat(order,:);
                end
                params.count = params.count + 1;
            end
        end
    end
    params.prior = params.prior/params.count;
    params.transmat = params.transmat/params.count;
    if discrete
        params.obsmat = params.obsmat/params.count;
    else
        params.mu = params.mu/params.count;
        params.sigma = params.sigma/params.count;
        params.mixmat = params.mixmat/params.count;
    end
end
    
% count = 0;
% if method == 1 || method == 4 % Using average (1) or weighted average (2) of last iteration of all the chains
%     % compute weight based on llh
%     if method == 4
%         llh_last = loglikelihood(end,:);
%         offset = min(llh_last);
%         weight = exp(llh_last - offset);
%         weight = weight/sum(weight);
%     else
%         weight = ones(1,num_trial);
%     end
%     for i = 1:num_trial
%         % ensure only use valid sample, which means the estimated hstate number matches true hstate
%         if hstate < max(params_set(i).params(end).assignment) % a valid assignment should only have value between 1 to hstate
%             warning('last iteration has different number of hidden state than truth. Sampling may not mixed');
%             %continue;
%         end
%         % find permutation on estimate state corresponding to true state;
%         % assuming substate number is the same across different trials
%         valid = params_set(i).params(end).assignment <= hstate;
%         perm_z = params_set(i).params(end).assignment(valid);
%         count = count + weight(i);
%         prior = params_set(i).params(end).prior(valid);
%         transmat = params_set(i).params(end).transmat(valid,valid);
%         mu = params_set(i).params(end).mu(:,valid,:);
%         covmat = params_set(i).params(end).sigma(:,:,valid,:);
%         mixmat = params_set(i).params(end).mixmat(valid,:);
%         if params_est.prior == 0
%             params_est.prior = zeros(size(prior));
%             params_est.transmat = zeros(size(transmat));
%             params_est.mu = zeros(size(mu));
%             params_est.sigma = zeros(size(covmat));
%             params_est.mixmat = zeros(size(mixmat));
%         end
%         params_est.prior(perm_z) = params_est.prior(perm_z) + weight(i)*prior/sum(prior);
%         params_est.transmat(perm_z,perm_z) = params_est.transmat(perm_z,perm_z) + weight(i)*bsxfun(@rdivide, transmat, sum(transmat,2));
%         % currently does not have a good way to make different trials be
%         % consistent in substate: perm_s (implemented but not verified in 'getParams')
%         % TO BE SAFE, USE UNI-MODAL GAUSSIAN AS EMISSION
%         params_est.mu(:,perm_z,:) = params_est.mu(:,perm_z,:) + weight(i)*mu;        
%         params_est.sigma(:,:,perm_z,:) = params_est.sigma(:,:,perm_z,:) + weight(i)*covmat;
%         params_est.mixmat(perm_z,:) = params_est.mixmat(perm_z,:) + weight(i)*mixmat;
%     end
%     params_est.prior = params_est.prior/count;
%     params_est.transmat = params_est.transmat/count;
%     params_est.mu = params_est.mu/count;
%     params_est.sigma = params_est.sigma/count;
%     params_est.mixmat = params_est.mixmat/count;
% 
% elseif method == 2 || method == 3 % Using average (2) or last iteration (3) of the chain which has the largest average likelihood 
%     llh_avg = mean(loglikelihood); % 1*num_trial vector
%     [~,opt_trial] = max(llh_avg);
%     if method == 2
%         idx = 1:num_iter;
%     else
%         idx = num_iter;
%     end
%     for i = idx
%         if method == 2
%             valid = params_set(opt_trial).params(i).assignment <= hstate;        
%             perm_z = params_set(opt_trial).params(i).assignment(valid);        
%         else
%             valid = 1:size(params_set(opt_trial).params(i).mu,2);
%             perm_z = valid;
%         end
%         count = count + 1;
%         prior = params_set(opt_trial).params(i).prior(valid);
%         transmat = params_set(opt_trial).params(i).transmat(valid,valid);
%         mu = params_set(opt_trial).params(i).mu(:,valid,:);
%         covmat = params_set(opt_trial).params(i).sigma(:,:,valid,:);
%         mixmat = params_set(opt_trial).params(i).mixmat(valid,:);
%         if params_est.prior == 0
%             params_est.prior = zeros(size(prior));
%             params_est.transmat = zeros(size(transmat));
%             params_est.mu = zeros(size(mu));
%             params_est.sigma = zeros(size(covmat));
%             params_est.mixmat = zeros(size(mixmat));
%         end
%         params_est.prior(perm_z) = params_est.prior(perm_z) + prior/sum(prior);
%         params_est.transmat(perm_z,perm_z) = params_est.transmat(perm_z,perm_z) + bsxfun(@rdivide, transmat, sum(transmat,2));
%         params_est.mu(:,perm_z,:) = params_est.mu(:,perm_z,:) + mu;        
%         params_est.sigma(:,:,perm_z,:) = params_est.sigma(:,:,perm_z,:) + covmat;
%         params_est.mixmat(perm_z,:) = params_est.mixmat(perm_z,:) + mixmat;
%     end
%     params_est.prior = params_est.prior/count;
%     params_est.transmat = params_est.transmat/count;
%     params_est.mu = params_est.mu/count;
%     params_est.sigma = params_est.sigma/count;
%     params_est.mixmat = params_est.mixmat/count;
%     
% % else
% 
% end

end