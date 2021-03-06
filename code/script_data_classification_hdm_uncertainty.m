%%%%%%%%%%% script to perform uncertainty analysis, assuming the results of
%%%%%%%%%%% corresponding experiment configuration is already available
close all; clearvars; dbstop if error;

%%%%%%%%%%% data loading
dataset_id = input('Enter dataset number:\n [1]UTD\n [2]G3D\n [3]MSRAction3D\n [4]Penn\n'); % choose which dataset to experiment with
if isempty(dataset_id) || dataset_id<1 || dataset_id>4
    error('Unsupported dataset. Enter a number between 1 and 4.');
end
savedir = '../results';
dir_base = '../data';
dataset_name = {'UTD','G3D','MSRA','Penn'};
load(fullfile(dir_base,dataset_name{dataset_id},'feature_locs.mat'),'feature','labels','Activity_label','nframe_count');
feature1_ = feature;
load(fullfile(dir_base,dataset_name{dataset_id},'feature_motions.mat'),'feature','lut_motions');
feature2_ = feature;
disp(strcat('Dataset selected:',dataset_name{dataset_id}));
inference_method = 1; % 1. blocked gibbs sampling; 2. mean field variational inference
gap = 10; % portion of increase in uncertainty
topology = [0 1 2 3 3 5 6 7 3 9 10 11 1 13 14 15 1 17 18 19];

%%%%%%%%%%% define constants
% dataset dependent configuration
if dataset_id == 1
    configuration_classification_UTD;
elseif dataset_id == 2
    configuration_classification_G3D;
elseif dataset_id == 3
    configuration_classification_MSRA;
elseif dataset_id == 4
    configuration_classification_Penn;
end
% pre-allocate variables
disp('Configuration completed.')

%%%%%%%%%%% main loop
for cv = 1:length(split_sub)
    fprintf('Split %d...\n',cv);
    
    % find index of data instance by subject
    [idx_train,idx_test] = find_idx(labels(:,2),split_sub(cv).train,split_sub(cv).validation);
    
    % data scaling and construct missing data
    nframe_train = zeros(length(idx_train),1);
    nframe_test = zeros(length(idx_test),1);
    
    % construct training and testing set and labels
    dataset_train = cell(num_class,1);
    dataset_test = cell(num_class,1);
    true_labels_train = [];
    true_labels_test = [];
    count_train = 0;
    count_test = 0;
    data_train = cell(1); % create a copy of the raw skeleton data for visualization purpose
    data_test = cell(1);
    for a = 1:num_class
        idx = find_idx(labels(idx_train,1),gestures(a));
        dataset_train{a} = feature(idx_train(idx));
        true_labels_train = [true_labels_train; a*ones(length(dataset_train{a}),1)];
        for i = 1:length(dataset_train{a})
            count_train = count_train + 1;
            nframe_train(count_train) = size(dataset_train{a}{i},2);
            tt = size(feature1_{idx_train(idx(i))},2);
            data_train{count_train} = [zeros(3,tt); feature1_{idx_train(idx(i))}];
        end
        idx = find_idx(labels(idx_test,1),gestures(a));
        dataset_test{a} = feature(idx_test(idx));
        true_labels_test = [true_labels_test; a*ones(length(dataset_test{a}),1)];
        for i = 1:length(dataset_test{a})
            count_test = count_test + 1;
            nframe_test(count_test) = size(dataset_test{a}{i},2);
            tt = size(feature1_{idx_test(idx(i))},2);
            data_test{count_test} = [zeros(3,tt); feature1_{idx_test(idx(i))}];
        end
    end
    
    nframe_train = nframe_train(1:count_train);
    nframe_test = nframe_test(1:count_test);
    
    % load experiment results
    if inference_method == 1
        %filename = fullfile(savedir,sprintf('HDM_Q%dto%d_hyper%d_adapt%d_stride%d_maxiter%d_config%d_PCA%d_cov%d_test%d_%s_%s_%s_dflag_truellh_mc%d_sub%d',Q,Q,useprior,adapt,stride,max_iter,config,(config>0)*round(100*portion),round(100*cov_prior_coeff_portion),test,cov_type,dataset_name{dataset_id},dura_type,MC,subsample));
        filename = fullfile(savedir,sprintf('HDM_Q%dto%d_hyper%d_adapt%d_stride%d_maxiter%d_config%d_PCA%d_cov%d_test%d_%s_%s_%s_dflag_truellh_mc%d_burn%d_sub%d',Q,Q,useprior,adapt,stride,max_iter,config,(config>0)*round(100*portion),round(100*cov_prior_coeff_portion),test,cov_type,dataset_name{dataset_id},dura_type,MC,burnin,subsample));
    else
        %filename = fullfile(savedir,sprintf('HDM_Q%dto%d_hyper%d_adapt%d_stride%d_maxiter%d_config%d_PCA%d_cov%d_test%d_%s_%s_%s_dflag_truellh_mc%d_sub%d_vi',Q,Q,useprior,adapt,stride,max_iter,config,(config>0)*round(100*portion),round(100*cov_prior_coeff_portion),test,cov_type,dataset_name{dataset_id},dura_type,MC,subsample));
        filename = fullfile(savedir,sprintf('HDM_Q%dto%d_hyper%d_adapt%d_stride%d_maxiter%d_config%d_PCA%d_cov%d_test%d_%s_%s_%s_dflag_truellh_mc%d_burn%d_sub%d_vi',Q,Q,useprior,adapt,stride,max_iter,config,(config>0)*round(100*portion),round(100*cov_prior_coeff_portion),test,cov_type,dataset_name{dataset_id},dura_type,MC,burnin,subsample));
    end
    load(filename,'hyperparams_set','params_set','acc','acc1','acc2','acc3','loglikelihood_all','loglikelihood'); % make sure Q is a scalar not vector
    if eval_train == 1
        load(filename,'acc_train','loglikelihood_all_train','loglikelihood_all_train_cv','loglikelihood_all_cv');
    end
    
    % compute the total uncertainty of training data
    if exist('loglikelihood_all_train_cv','var'), loglikelihood_all_train = loglikelihood_all_train_cv{cv}; end
    [total_uncertainty_train, total_cov_avg_train, probs_train, mean_cov_train, cov_mean_train, N_train, covs_train] = est_uncertainty(loglikelihood_all_train,nframe_train);

    % compute the total uncertainty of testing data
    % first term: data uncertainty E[V[y|X]]: num_class x num_class x N    
    if exist('loglikelihood_all_cv','var'), loglikelihood_all = loglikelihood_all_cv{cv}; end
    [total_uncertainty, total_cov_avg, probs, mean_cov, cov_mean, N, covs] = est_uncertainty(loglikelihood_all,nframe_test);    
    % another way of computing uncertainty
    %mean_var = mean(probs.*(1-probs),3); % N_heldout x num_class
    %var_mean = var(probs,[],3); % N_heldout x num_class
    %total_var = mean_var + var_mean; % N_heldout x num_class  %%%% this one is problematic 
    %total_uncertainty = sum(total_var,2); % N_heldout x 1
    % assert 0
    %norm(total_uncertainty_-total_uncertainty)    
    
    %%%%%%%%%%%%%%%%% FIGURE 0:
    figure; % training data has overall much smaller uncertainty than testing data
    plot(1:N_train,sort(total_uncertainty_train),'r',1:N,sort(total_uncertainty),'b');
    legend({'train','test'})
    ylabel('uncertainty value')
    xlabel('instance')
    title(dataset_name{dataset_id})
    
    %%%%%%%%%%%%%%%%% FIGURE 0.1:
    if dataset_id == 1
        displayrange = [1 N_train+50 0 10];
    elseif dataset_id == 2
        displayrange = [1 N_train+50 0 5];
    end
    % compute the data uncertainty produced by each model
    data_uncertainty_train_sort = zeros(N_train,MC);
    for n = 1:N_train
        for m = 1:MC
            data_uncertainty_train_sort(n,m) = trace(squeeze(covs_train(:,:,n,m))); % covs_train: num_class x num_class x N_train x MC
        end
    end
    % sort all columns by the total_uncertainty
    [~,idx] = sort(total_uncertainty_train); % N_train x 1
    true_labels_train_sort = true_labels_train(idx);
    data_uncertainty_train_sort = data_uncertainty_train_sort(idx,:);
    data_uncertainty_sort = zeros(N,MC);
    for n = 1:N
        for m = 1:MC
            data_uncertainty_sort(n,m) = trace(squeeze(covs(:,:,n,m))); % covs: num_class x num_class x N_test x MC
        end
    end
    % sort all columns by the total_uncertainty
    [~,idx] = sort(total_uncertainty);
    true_labels_test_sort = true_labels_test(idx);
    data_uncertainty_sort = data_uncertainty_sort(idx,:);

    %%%%%%%%%%%%%%%%%%% EXPERIMENT: use training data uncertainty of each individual
    %%%%%%%%%%%%%%%%%%% model to weight their contribution on the testing
    %%%%%%%%%%%%%%%%%%% data
    data_uncertainty_train_by_model = mean(data_uncertainty_train_sort); % 1 x MC
    weights_model = data_uncertainty_train_by_model/sum(data_uncertainty_train_by_model);
    
    weights_model_topK_train = zeros(MC,MC); % weights according to the training data
    [~,idx_model_train] = sort(data_uncertainty_train_by_model);    
    for m = 1:MC
        weights_model_topK_train(m,idx_model_train(1:m)) = data_uncertainty_train_by_model(idx_model_train(1:m))/sum(data_uncertainty_train_by_model(idx_model_train(1:m)));%1/m;
    end
    
    weights_model_topK = zeros(N,MC,MC); % now the weights are different for different testing data
    for n = 1:N % for each testing data
        [~,idx_model] = sort(data_uncertainty_sort(n,:));
        for m = 1:MC
            weights_model_topK(n,m,idx_model(1:m)) = data_uncertainty_sort(n,idx_model(1:m))/sum(data_uncertainty_sort(n,idx_model(1:m)));%1/m;
        end
    end
    
    data_uncertainty_train_by_class_sort = zeros(MC,num_class); % use this to generate weights: MC x class
    for m = 1:MC % select 10 samples from all MC samples
        for act = 1:num_class        
            data_uncertainty_train_by_class_sort(m,act) = mean(data_uncertainty_train_sort(true_labels_train_sort==act,m));
        end
    end
    weights = bsxfun(@rdivide, 1./data_uncertainty_train_by_class_sort, sum(1./data_uncertainty_train_by_class_sort)); % each column sum to 1: MC x num_class
    % weighted probability
    scores = zeros(N,num_class);
    scores_model = zeros(N,num_class);
    scores_model_topK_train = zeros(MC,N,num_class);
    scores_model_topK = zeros(MC,N,num_class);
    for n = 1:N
        probs_ = squeeze(probs(n,:,:)); % num_class x MC
        scores(n,:) = sum(probs_ .* weights',2)'; % 1 x num_class, does not sum to 1
        scores_model(n,:) = (probs_ * weights_model')';
        for m = 1:MC
            scores_model_topK_train(m,n,:) = (probs_ * weights_model_topK_train(m,:)')';
            scores_model_topK(m,n,:) = (probs_ * squeeze(weights_model_topK(n,m,:)))';
        end
    end
    acc_weight = compute_accuracy(scores,true_labels_test,num_class);
    acc_weight_model = compute_accuracy(scores_model,true_labels_test,num_class);
    acc_weight_model_topK_train = zeros(MC,1);
    acc_weight_model_topK = zeros(MC,1);
    for m = 1:MC
        acc_weight_model_topK_train(m) = compute_accuracy(squeeze(scores_model_topK_train(m,:,:)),true_labels_test,num_class);
        acc_weight_model_topK(m) = compute_accuracy(squeeze(scores_model_topK(m,:,:)),true_labels_test,num_class);
    end
    acc_original = compute_accuracy(mean(loglikelihood_all,3),true_labels_test,num_class); % mean(probs,3)
    fprintf('Original:%.4f,Weighted:%.4f,Weighted by model:%.4f\n',acc_original,acc_weight,acc_weight_model);
    for m = 1:MC
        fprintf('Weighted by top %d training models:%.4f, testing models:%.4f\n',m,acc_weight_model_topK_train(m),acc_weight_model_topK(m));
    end
    
    %%%%%%%%%%%%%%%%% FIGURE 1.1: Average Covariance (the first term of the total variance, this one can be generated w/o needing to know ground truth labels)
    fig = figure('rend','painters','pos',[500 100 800 600]); %1200 900 width, height
    drawcm(mean(mean_cov,3),'Labels',Activity_label(gestures),'Toggle',false,'Normalize',false,'Legend',false);
    title('Average E[V[y|X]]'); % (data)
    colorbar;
%     saveas(gcf,fullfile(savedir,strcat('cov_data_',dataset_name{dataset_id},sprintf('_maxiter%d_mc%d_infer%d.png',max_iter,MC,inference_method))));

    %%%%%%%%%%%%%%%%% FIGURE 1.2: Model Covariance (the second term of the total variance, this one can be generated w/o needing to know ground truth labels)
    fig = figure('rend','painters','pos',[500 100 800 600]); %1200 900 width, height
    drawcm(mean(cov_mean,3),'Labels',Activity_label(gestures),'Toggle',false,'Normalize',false,'Legend',false);
    title('Average V[E[y|X]]'); % (model uncertainty)
    colorbar;
%     saveas(gcf,fullfile(savedir,strcat('cov_model_',dataset_name{dataset_id},sprintf('_maxiter%d_mc%d_infer%d.png',max_iter,MC,inference_method))));

    %%%%%%%%%%%%%%%%% FIGURE 2: total covariance
    fig = figure('rend','painters','pos',[500 100 800 600]); %1200 900 width, height
    drawcm(total_cov_avg,'Labels',Activity_label(gestures),'Toggle',false,'Normalize',false,'Legend',false);
    title('Average V[y|x]'); % (total covariance)
    colorbar;
%     saveas(gcf,fullfile(savedir,strcat('cov_total_',dataset_name{dataset_id},sprintf('_maxiter%d_mc%d_infer%d.png',max_iter,MC,inference_method))));

    %%%%%%%%%%%%%%%%% FIGURE 3: Probability of each individual sequence (this one need to know ground truth labels so that the data is sorted to reflect diagonal structure)
    figure;
    imagesc(mean(probs,3)) % see which one is more confused, looks like confusion matrix
    title('average probability');
    xlabel('class')
    ylabel('instance')
%     saveas(gcf,fullfile(savedir,strcat('probs_',dataset_name{dataset_id},sprintf('_maxiter%d_mc%d_infer%d.png',max_iter,MC,inference_method))));
    
    %%%%%%%%%%%%%%%%% FIGURE 4: Average Probability of sequences within each class (this one need to know ground truth labels so that the data is sorted to reflect diagonal structure)
    probs_classwise = zeros(num_class,num_class);
    probs_classwise_std = probs_classwise;
    probs_classwise_train = zeros(num_class,num_class);
    probs_classwise_train_std = probs_classwise_train;
    probs_avg = mean(probs,3); % N x num_class
    probs_avg_train = mean(probs_train,3); % N x num_class
    entropy_avg = -sum(probs_avg.*log2(probs_avg+eps),2);
    entropy_avg_train = -sum(probs_avg_train.*log2(probs_avg_train+eps),2);
    for a = 1:num_class
        probs_classwise(a,:) = mean(probs_avg(true_labels_test==a,:));
        probs_classwise_std(a,:) = std(probs_avg(true_labels_test==a,:));
        probs_classwise_train(a,:) = mean(probs_avg_train(true_labels_train==a,:));
        probs_classwise_train_std(a,:) = std(probs_avg_train(true_labels_train==a,:));
        entropy_classwise(a,:) = mean(entropy_avg(true_labels_test==a,:));
        entropy_classwise_std(a,:) = std(entropy_avg(true_labels_test==a,:));
        entropy_classwise_train(a,:) = mean(entropy_avg_train(true_labels_train==a,:));
        entropy_classwise_train_std(a,:) = std(entropy_avg_train(true_labels_train==a,:));
    end
    figure;
    drawcm(probs_classwise,'Labels',Activity_label(gestures),'Toggle',false,'Normalize',false,'Legend',false);
    title('average within-class probability');
    
    portion_uncertainty = 10:gap:100;
    % sorted training accuracy
    loglikelihood_train = zeros(size(loglikelihood_all_train,1),size(loglikelihood_all_train,2));
    for i = 1:size(loglikelihood_all_train,1)
        for j = 1:size(loglikelihood_all_train,2)
            llh = loglikelihood_all_train(i,j,:);
            idx = (~isnan(llh))&(~isinf(llh));
            loglikelihood_train(i,j) = mean(llh(idx));
        end
    end
    [~,cmatrix_train,cmatrix_norm_train,predict_labels_train] = compute_accuracy(loglikelihood_train,true_labels_train,num_class);    
    [total_uncertainty_sort_train,idx,accuracy_model_avg_portion_train,accuracy_model_avg_portion_within_train(:,cv),labels_model_avg_sort_train,true_labels_train_sort] = est_acc_uncertainty(predict_labels_train,true_labels_train,total_uncertainty_train,portion_uncertainty);
    
    % sorted testing accuracy
    loglikelihood = zeros(size(loglikelihood_all,1),size(loglikelihood_all,2));
    for i = 1:size(loglikelihood_all,1)
        for j = 1:size(loglikelihood_all,2)
            llh = loglikelihood_all(i,j,:);
            idx = (~isnan(llh))&(~isinf(llh));
            loglikelihood(i,j) = mean(llh(idx));
        end
    end
    [~,cmatrix,cmatrix_norm,predict_labels] = compute_accuracy(loglikelihood,true_labels_test,num_class); 
    [total_uncertainty_sort,idx,accuracy_model_avg_portion,accuracy_model_avg_portion_within(:,cv),labels_model_avg_sort,true_labels_test_sort] = est_acc_uncertainty(predict_labels,true_labels_test,total_uncertainty,portion_uncertainty);    
    
    %%%%%%%%%%%%%%%%% FIGURE 5: error rate vs. uncertainty
    figure;
    %plot(1-accuracy_model_avg_portion,'LineWidth',2)
    plot(1:length(portion_uncertainty),1-accuracy_model_avg_portion_train,'r',1:length(portion_uncertainty),1-accuracy_model_avg_portion,'b')
    legend({'train','test'})
    title(dataset_name{dataset_id})
    xlabel('Percentage of lowest uncertainty')
    ax = gca;
    ax.XTick = 1:max_iter;
    ax.XTickLabel = portion_uncertainty;
    ylabel('Error rate')
%     saveas(gcf,fullfile(savedir,strcat('uncertainty_',dataset_name{dataset_id},sprintf('_maxiter%d_mc%d_infer%d.png',max_iter,MC,inference_method))));

    figure;
    %bar(1-accuracy_model_avg_portion_within);
    bar([1-accuracy_model_avg_portion_within_train(:,cv) 1-accuracy_model_avg_portion_within(:,cv)]);
    legend({'train','test'})
    title(dataset_name{dataset_id})
    xlabel('Level of uncertainty')
    ax = gca;
    ax.XTick = 1:max_iter;
    ax.XTickLabel = portion_uncertainty;
    ylabel('Error rate')
%     saveas(gcf,fullfile(savedir,strcat('uncertainty_within_',dataset_name{dataset_id},sprintf('_maxiter%d_mc%d_infer%d.png',max_iter,MC,inference_method))));

    %%%%%%%%%%%%%%%%% FIGURE 6: confusion matrix
    fig = figure('rend','painters','pos',[500 100 800 600]); %1200 900 width, height
    drawcm(cmatrix,'Labels',Activity_label(gestures),'Toggle',false);
    title('confusion matrix')
    colorbar;
%     saveas(gcf,fullfile(savedir,strcat('cm_',dataset_name{dataset_id},sprintf('_maxiter%d_mc%d_infer%d.png',max_iter,MC,inference_method))));    
    
    %%%%%%%%%%%%%%%%% FIGURE 6.1 samples of data with different uncertainty
    % most certain top 10
    % action label
    step_t = 3;
    id = 1;
    if size(data_test{idx(id)},1) == 3*length(topology) %%%%%%% only for 3D dataset
        figure;
        for t = 1:step_t:size(data_test{idx(id)},2)            
            drawskt3(data_test{idx(id)}(1:3:end,t),data_test{idx(id)}(3:3:end,t),data_test{idx(id)}(2:3:end,t),1:20,topology,'MarkerSize',15,'LineWidth',5,'jointID',false)
            grid off
            axis off
            title('')
    %         saveas(gcf,fullfile(savedir,sprintf('skel_%s_%d_t%d.png',dataset_name{dataset_id},id,t)),'png');
        end
    end
    % least certain
    if dataset_id == 3 && inference_method == 2
        topology_ = [0 1 2 3 3 5 6 7 3 9 10 11 1 17 14 15 1 13 18 19];
    else
        topology_ = topology;
    end
    id = length(idx)-7*(dataset_id==1)-5*(dataset_id==3);
    if size(data_test{idx(id)},1) == 3*length(topology) %%%%%%% only for 3D dataset
        figure;
        for t = 1:step_t:size(data_test{idx(id)},2)            
            drawskt3(data_test{idx(id)}(1:3:end,t),data_test{idx(id)}(3:3:end,t),data_test{idx(id)}(2:3:end,t),1:20,topology_,'MarkerSize',15,'LineWidth',5,'jointID',false)
            grid off
            axis off
            title('')
    %         saveas(gcf,fullfile(savedir,sprintf('skel_%s_%d_t%d.png',dataset_name{dataset_id},id,t)),'png');
        end
    end
    
    %%%%%%%%%%%%%%%%% FIGURE 7: check the probability bar chart corresponding to specific testing sample
    % low uncertainty testing sample
    figure;
    id = 1;
    bar(1:num_class,probs_avg(idx(id),:));
    xlabel('Class')
    ylabel('Probability')
    title(sprintf('Actual: %d. Predicted: %d.',true_labels_test_sort(id),labels_model_avg_sort(id)));
    title(sprintf('Actual: %s. Predicted: %s.',Activity_label{gestures(true_labels_test_sort(id))},Activity_label{gestures(labels_model_avg_sort(id))}));
    axis([0 num_class+1 0 1])
%     saveas(gcf,fullfile(savedir,sprintf('prob_%s_infer%d_%d.png',dataset_name{dataset_id},inference_method,id)),'png');
    % high uncertainty testing sample
    figure;
    id = length(idx)-7*(dataset_id==1)-5*(dataset_id==3);
    bar(1:num_class,probs_avg(idx(id),:));
    xlabel('Class')
    ylabel('Probability')
    title(sprintf('Actual: %d. Predicted: %d.',true_labels_test_sort(id),labels_model_avg_sort(id)));
    title(sprintf('Actual: %s. Predicted: %s.',Activity_label{gestures(true_labels_test_sort(id))},Activity_label{gestures(labels_model_avg_sort(id))}));
    axis([0 num_class+1 0 1])
%     saveas(gcf,fullfile(savedir,sprintf('prob_%s_infer%d_%d.png',dataset_name{dataset_id},inference_method,id)),'png');
    
    %%%%%%%%%%%%%%%%% FIGURE 8: histogram bin plot showing error rate versus
    figure;
    h = bar(total_uncertainty_sort);

    %%%%%%%%%%%%%%%%% FIGURE 9: class-wise uncertainty analysis
    % 1. average uncertainty of individual classes (testing data)
    testing = 1;
    total_uncertainty_class = zeros(num_class,2); % mean and std in each row
    accuracy_class = zeros(num_class,1);
    for a = 1:num_class        
        if testing == 1
            idx_a = find(true_labels_test==a);
            total_uncertainty_class(a,1) = mean(total_uncertainty(idx_a));
            total_uncertainty_class(a,2) = std(total_uncertainty(idx_a));
            accuracy_class(a) = cmatrix_norm(a,a);
        else
            idx_a = find(true_labels_train==a);
            total_uncertainty_class(a,1) = mean(total_uncertainty_train(idx_a));
            total_uncertainty_class(a,2) = std(total_uncertainty_train(idx_a));
            accuracy_class(a) = cmatrix_norm_train(a,a);
        end        
%         labels_model_avg_sort
    end
    fig = figure('rend','painters','pos',[100 100 num_class*30 300]);
    set(fig,'defaultAxesColorOrder',[[0 0 0]; [1 0 0]]);
    if testing == 1
        [~,idx_class] = sort(total_uncertainty_class(:,1),'ascend');
%         [~,idx_class] = sort(entropy_classwise,'ascend');
%         [~,idx_class] = sort(-diag(probs_classwise),'ascend');
    else
        [~,idx_class] = sort(total_uncertainty_class(:,1),'ascend');
%         [~,idx_class] = sort(entropy_classwise_train,'ascend');
%         [~,idx_class] = sort(-diag(probs_classwise_train),'ascend');
    end
    h = barwitherr(total_uncertainty_class(idx_class,2), ...
        total_uncertainty_class(idx_class,1));% Plot with errorbars 
    ax = gca;
    ax.XTick = 1:num_class;
    ax.XTickLabel = Activity_label(gestures(idx_class));
    rotateXLabels(ax,30);
%     xlabel('Class');
    ylabel('Uncertainty')%ylabel('Entropy')%ylabel('Probability')%
%     legend({'Uncertainty','Error'})
    axis([0 num_class+1 0 1])%log2(num_class)
    yyaxis right
    plot(accuracy_class(idx_class),'r','LineWidth',2)
    axis([0 num_class+1 0 1])%
    ylabel('Accuracy')
    if testing == 1
        title(dataset_name{dataset_id})
    else
        title(strcat(dataset_name{dataset_id},' train'))
    end
    saveas(ax,fullfile(savedir,sprintf('uncertainty_class_%s_infer%d.png',dataset_name{dataset_id},inference_method)),'png');
    % print out the class name
    fprintf('From least uncertain to most uncertain\n')
    for a = 1:num_class
        fprintf('Class %d, %s\n',idx_class(a),Activity_label{gestures(idx_class(a))});
    end
    
    % 1. average uncertainty of individual classes (training)
    
    % 2. average accuracy of the class
    
    % confusion condition
    
    %%%%%%%%%%%%%%%%% FIGURE 10: cross-entropy vs. entropy
    figure;
    cross_entropy = zeros(N,1);
    for n = 1:N
        cross_entropy(n) = -log(probs_avg(n,true_labels_test(n))+eps);        
    end
    entropy = -sum(probs_avg.*log(probs_avg+eps),2);
    %[entropy_sort,idx] = sort(entropy);
    %plot(entropy_sort,cross_entropy(idx));%bar(entropy,cross_entropy);
    [cross_entropy_sort,idx] = sort(cross_entropy);
    plot(cross_entropy_sort,entropy(idx));%bar(entropy,cross_entropy);
    xlabel('entropy')
    ylabel('cross entropy')
    
    %%%%%%%%%%%%%%%%% FIGURE 4.1: probability map, sort by uncertainty value
    % num_class x num_class
%     fig = figure('rend','painters','pos',[500 100 800 600]);
%     drawcm(probs_classwise_train(idx_class,idx_class),'Labels',Activity_label(gestures(idx_class)),'Toggle',false,'Normalize',false,'Legend',false);
%     title('average within-class probability sorted by uncertainty (train)');
%     colorbar;
%     
%     fig = figure('rend','painters','pos',[500 100 800 600]);
%     drawcm(probs_classwise(idx_class,idx_class),'Labels',Activity_label(gestures(idx_class)),'Toggle',false,'Normalize',false,'Legend',false);
%     title('average within-class probability sorted by uncertainty');
%     colorbar;
    
    % 6.1 confusion matrix, sort by uncertainty value
    fig = figure('rend','painters','pos',[500 100 800 600]); %1200 900 width, height
    drawcm(cmatrix(idx_class,idx_class),'Labels',Activity_label(gestures(idx_class)),'Toggle',false);
    title('confusion matrix sorted by uncertainty')
    colorbar;
    
    fig = figure('rend','painters','pos',[500 100 800 600]); %1200 900 width, height
    drawcm(cmatrix_train(idx_class,idx_class),'Labels',Activity_label(gestures(idx_class)),'Toggle',false);
    title('confusion matrix sorted by uncertainty (train)')
    colorbar;
    
    % 2.1
    fig = figure('rend','painters','pos',[500 100 800 600]); %1200 900 width, height
    drawcm(total_cov_avg(idx_class,idx_class),'Labels',Activity_label(gestures(idx_class)),'Toggle',false,'Normalize',false,'Legend',false);
    title('Average V[y|x] sorted by uncertainty'); % (total covariance)
    colorbar;
    
    fig = figure('rend','painters','pos',[500 100 800 600]); %1200 900 width, height
    drawcm(total_cov_avg_train(idx_class,idx_class),'Labels',Activity_label(gestures(idx_class)),'Toggle',false,'Normalize',false,'Legend',false);
    title('Average V[y|x] sorted by uncertainty (train)'); % (total covariance train)
    colorbar;
    
end
% close all;
% check out on average, what is the training/validation scaling factor
figure;
bar([1-mean(accuracy_model_avg_portion_within_train,2) 1-mean(accuracy_model_avg_portion_within,2)]);
legend({'train avg','test avg'})
title(dataset_name{dataset_id})
xlabel('Level of uncertainty')
ax = gca;
ax.XTick = 1:max_iter;
ax.XTickLabel = portion_uncertainty;
ylabel('Error rate')