%% script to perform training and testing using HDM on sequential features
close all; clearvars; dbstop if error;

%% add dependency
addpath('utils');
addpath('../tools/bnt');
addpath(genpathKPM(pwd));

%% data loading
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

%% set experiment configuration
reload_model = false;
save_for_inference = false;
save_burnin = true;
llh_flag = false; % whether compute llh when performing Gibbs sampling

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
acc = zeros(1,length(split_sub));
predict_labels = cell(1,length(split_sub));
acc1 = zeros(1,length(split_sub)); 
acc2 = zeros(1,length(split_sub)); 
acc3 = zeros(1,length(split_sub)); 
acc_train = acc;
cmatrix = zeros(num_class,num_class,length(split_sub)); 
cmatrix_train = cmatrix;
cmatrix_norm = zeros(num_class,num_class,length(split_sub)); 
cmatrix_norm_train = cmatrix_norm;
time = zeros(1,length(split_sub));
hyperparams_set = repmat(struct('hyperparams',[]),num_class,length(split_sub));
params_set = repmat(struct('params',[]),num_class,length(split_sub));
params_samples_all = repmat(struct('params_samples',[]),num_class,length(split_sub));
params_samples_all_burnin = repmat(struct('params_samples',[]),num_class,length(split_sub));
llh_train = cell(num_class,length(split_sub));
% detailed info
llhptrace_verbose = cell(num_class,length(split_sub));
LLtrace = cell(num_class,length(split_sub));
llhptrace = cell(num_class,length(split_sub));
llhhtrace = cell(num_class,length(split_sub));
hyperparams_trace = cell(num_class,length(split_sub));
loglikelihood_all_cv = cell(length(split_sub),1);
loglikelihood_all_train_cv = cell(length(split_sub),1);

disp('Configuration completed.')

%%%%%%%%%%% main loop
for cv = 1:length(split_sub)
    fprintf('Split %d...\n',cv);
    
    % find index of data instance by subject
    [idx_train,idx_test] = find_idx(labels(:,2),split_sub(cv).train,split_sub(cv).validation);
    
    % data scaling and construct missing data
    feature_scaling; 
    disp('Feature scaling is completed.')
    % after this, the variable 'feature' is updated
    L = max(nframe_count);
    O = size(feature{idx_train(1)},1);
    
    % construct training and testing set and labels
    dataset_train = cell(num_class,1);
    dataset_test = cell(num_class,1);
    true_labels_train = [];
    true_labels_test = [];
    for a = 1:num_class
        idx = find_idx(labels(idx_train,1),gestures(a));
        dataset_train{a} = feature(idx_train(idx));
        true_labels_train = [true_labels_train; a*ones(length(dataset_train{a}),1)];
        idx = find_idx(labels(idx_test,1),gestures(a));
        dataset_test{a} = feature(idx_test(idx));
        true_labels_test = [true_labels_test; a*ones(length(dataset_test{a}),1)];
    end
    
    % creat a copy of training and testing data for test purpose
    data_train = [];
    data_test = [];
    for a = 1:num_class
        data_train = [data_train; dataset_train{a}];
        data_test = [data_test; dataset_test{a}];
    end
    
    % training and testing likelihood: different cv fold may have different
    % training, testing instances, so this part is not initialized outside
    loglikelihood_train = zeros(length(data_train),num_class);
    loglikelihood = zeros(length(data_test),num_class);    
    loglikelihood_all = zeros(length(data_test),num_class,MC);    
    loglikelihood_all_train = zeros(length(data_train),num_class,MC);
    loglikelihood1 = zeros(length(data_test),num_class);
    loglikelihood2 = zeros(length(data_test),num_class);
    loglikelihood3 = zeros(length(data_test),num_class);
    
    % estimate initial spatio hyperparameters
    [mu0, sigma0] = est_hyper_emis(feature(idx_train),cov_type,sum(nframe_count(idx_train)));
    
    %% different state numbers
    for it = 1:length(Q)
        tt = tic;
        fprintf('-Iteration Q=%d\n',Q(it))
        % no matter whether using hyperparam or not, provide an initialization
        hyperparams = ini_hyperparams(Q(it),O,L,dura_prior_coeff,useprior,cov_prior_coeff,dura_type);
        hyperparams.emis.kappa = 1;
        hyperparams.emis.mu = mu0;
        hyperparams.emis.S = sigma0;
        
        % load pre-trained model
        if reload_model
            load(fullfile(savedir,sprintf('HDM_Q%dto%d_hyper%d_adapt%d_stride%d_maxiter%d_config%d_PCA%d_cov%d_test%d_%s_%s_%s_dflag_truellh_mc0_burn0_sub%d',Q(it),Q(it),useprior,adapt,stride,max_iter,config,(config>0)*round(100*portion),round(100*cov_prior_coeff_portion),test,cov_type,dataset_name{dataset_id},dura_type,subsample)),'hyperparams_set','params_set'); % make sure Q is a scalar not vector
        end
        
        if save_for_inference
            % hyperparameters for VI: only used the last fold of split_sub
            alpha = zeros(num_class,1);
            alpha_0 = zeros(Q(it),num_class);
            beta_0 = zeros(Q(it),num_class);
            mu_0 = zeros(O,num_class);
            sigma_0 = zeros(O,O,num_class);
            kappa_0 = zeros(num_class,1);
            nu_0 = zeros(num_class,1);
        end
        
        for a = 1:num_class
            rng default;
            cov_prior = repmat(cov_prior_coeff*eye(O), [1 1 M*Q(it)]);
            dura_prior = dura_prior_coeff*ones(Q(it),L); % repmat(1./(1:L),Q(it),1); %
            %%%%%%%%%%%%%%  TRAINING: train the model by MLE on hyperparameters
            if FB == 0
                if ~reload_model
                    [hyperparams_set(a,cv).hyperparams,params_set(a,cv).params,LLtrace{a,cv},llhptrace_verbose{a,cv},hyperparams_trace{a,cv},llhptrace{a,cv}, llhhtrace{a,cv}] = ...
                        bhsmm_eb2(dataset_train{a}(1:subsample:end), Q(it), L, hyperparams, 'max_iter', max_iter, 'max_iter_em', max_iter_em, ...
                        'cov_type', cov_type, 'cov_prior', cov_prior, 'dura_prior', dura_prior, 'hstate', cell(1), 'useprior', useprior, 'tol', 1e-4, 'dura_type', dura_type); %_gamma
                    if useprior == 2
                        params_set(a,cv).params = hsmm_em_new(dataset_train{a}(1:subsample:end), L, params_set(a,cv).params.prior, params_set(a,cv).params.transmat, params_set(a,cv).params.duramat, params_set(a,cv).params.mu, params_set(a,cv).params.sigma, hyperparams_set(a,cv).hyperparams, 'max_iter', max_iter_em, 'cov_type', cov_type, 'cov_prior', cov_prior, 'dura_type', dura_type); %_gamma
                    end
                end
                params = params_set(a,cv).params;                
            end
            %%%%%%%%%%%%%%  TESTING ON TESTING SET: compute likelihood
            if MC == 0 && FB == 0 %%%%%%%%%%%%% empirical Bayesian approach with MAP approx: on transition parameters
                for n = 1:length(data_test)
                    if adapt == 1
                        params = hsmm_em_new(data_test(n), L, params_set(a,cv).params.prior, params_set(a,cv).params.transmat, params_set(a,cv).params.duramat, params_set(a,cv).params.mu, params_set(a,cv).params.sigma, hyperparams_set(a,cv).hyperparams, 'max_iter', max_iter_em, 'cov_type', cov_type, 'cov_prior', cov_prior, 'verbose', false, 'clamped', 1, 'dura_type', dura_type); %_gamma
                    end
                    loglikelihood(n,a) = compute_llh_evidence_HSMM(data_test(n), params, L, 'dura_type', dura_type);
                end
                %%%%%%%%%%%%%%  TESTING ON TRAINING SET: compute likelihood
                if eval_train == 1
                    for n = 1:length(data_train)
                        loglikelihood_train(n,a) = compute_llh_evidence_HSMM(data_train(n), params, L, 'dura_type', dura_type);
                    end
                end
            elseif MC > 0 && FB == 0 %%%%%%%%%%%%% empirical Bayesian approach with MC approx: on transition parameters
                % single chain based inference
                %[params_samples,llh_train{a,cv}] = sample_params_hsmm(dataset_train{a}(1:subsample:end), params, hyperparams_set(a,cv).hyperparams, Q(it), O, L, MC, dura_type, burnin);
                % multiple chains based inference
                [params_samples,llh_train{a,cv},params_samples_burnin] = sample_params_hsmm2(dataset_train{a}(1:subsample:end), params, hyperparams_set(a,cv).hyperparams, Q(it), O, L, MC, dura_type, burnin, llh_flag);
                fprintf('class %d sampling completed\n',a);
                params_samples_all(a,cv).params_samples = params_samples;
                params_samples_all_burnin(a,cv).params_samples = params_samples_burnin;
                for n = 1:length(data_test)
                    llh = zeros(MC,1); % from this we can compute the uncertainty of the model
                    for m = 1:MC
                        llh(m) = compute_llh_evidence_HSMM(data_test(n), params_samples(m).params, L, 'dura_type', dura_type); % use this if you use 'sample_params_hsmm2'
                    end
                    loglikelihood_all(n,a,:) = llh;
                    idx = (~isnan(llh))&(~isinf(llh));
                    loglikelihood(n,a) = mean(llh(idx));
                    loglikelihood1(n,a) = min(llh(idx));
                    loglikelihood2(n,a) = max(llh(idx));
                    loglikelihood3(n,a) = median(llh(idx));
                end
                %%%%%%%%%%%%%%  TESTING ON TRAINING SET: compute likelihood
                if eval_train == 1
                    for n = 1:length(data_train)
                        llh = zeros(MC,1); % from this we can compute the uncertainty of the model
                        for m = 1:MC
                            llh(m) = compute_llh_evidence_HSMM(data_train(n), params_samples(m).params, L, 'dura_type', dura_type); % use this if you use 'sample_params_hsmm2'
                        end
                        loglikelihood_all_train(n,a,:) = llh;
                        idx = (~isnan(llh))&(~isinf(llh));
                        loglikelihood_train(n,a) = mean(llh(idx));                        
                    end
                end
            elseif FB == 1 %%%%%%%%%%%% full Bayesian approach
                [params_samples, LLtrace] = bhsmm_fb(dataset_train{a}(1:subsample:end),Q(it),O,L,MC,'max_iter', 20,'cov_type', cov_type, 'cov_prior', cov_prior, 'dura_type',dura_type,'dura_prior',dura_prior);
                plot(LLtrace);
                for n = 1:length(data_test)
                    llh = zeros(MC,1);
                    for m = 1:MC
                        llh(m) = compute_llh_evidence_HSMM(data_test(n), params_samples(m).params, L, 'dura_type', dura_type);
                    end
                    loglikelihood_all(n,a,:) = llh;
                    idx = (~isnan(llh))&(~isinf(llh));
                    loglikelihood(n,a) = mean(llh(idx));
                    loglikelihood1(n,a) = min(llh(idx));
                    loglikelihood2(n,a) = max(llh(idx));
                    loglikelihood3(n,a) = median(llh(idx));
                end
                %%%%%%%%%%%%%%  TESTING ON TRAINING SET: compute likelihood
                if eval_train == 1
                    for n = 1:length(data_train)
                        llh = zeros(MC,1); % from this we can compute the uncertainty of the model
                        for m = 1:MC
                            llh(m) = compute_llh_evidence_HSMM(data_train(n), params_samples(m).params, L, 'dura_type', dura_type); % use this if you use 'sample_params_hsmm2'
                        end
                        loglikelihood_all_train(n,a,:) = llh;
                        idx = (~isnan(llh))&(~isinf(llh));
                        loglikelihood_train(n,a) = mean(llh(idx));                        
                    end
                end
            end
            fprintf('class %d llh completed\n',a);
        end
        
        % llh of all data at current fold
        loglikelihood_all_cv{cv} = loglikelihood_all;
        loglikelihood_all_train_cv{cv} = loglikelihood_all_train;

        % training results
        if eval_train == 1
            [acc_train(cv),cmatrix_train(:,:,cv),cmatrix_norm_train(:,:,cv)] = compute_accuracy(loglikelihood_train,true_labels_train,num_class);
            figure; drawcm(cmatrix_norm_train(:,:,cv),'Labels',Activity_label(gestures)); % ,'Angle',-30
        end
        
        % testing results
        [acc(cv),cmatrix(:,:,cv),cmatrix_norm(:,:,cv),predict_labels{cv}] = compute_accuracy(loglikelihood,true_labels_test,num_class);
        figure; drawcm(cmatrix_norm(:,:,cv),'Labels',Activity_label(gestures)); % ,'Angle',-30
        title(sprintf('accuracy:%f',acc(cv)));
        acc1(cv) = compute_accuracy(loglikelihood1,true_labels_test,num_class);
        acc2(cv) = compute_accuracy(loglikelihood2,true_labels_test,num_class);
        acc3(cv) = compute_accuracy(loglikelihood3,true_labels_test,num_class);
        
        %% record time
        time(cv) = toc(tt);
        
        % duplicate saving for each cv for timely saving results        
        filename = fullfile(savedir,sprintf('HDM_Q%dto%d_hyper%d_adapt%d_stride%d_maxiter%d_config%d_PCA%d_cov%d_test%d_%s_%s_%s_dflag_truellh_mc%d_burn%d_sub%d',Q(it),Q(it),useprior,adapt,stride,max_iter,config,(config>0)*round(100*portion),round(100*cov_prior_coeff_portion),test,cov_type,dataset_name{dataset_id},dura_type,MC,burnin*(MC>0),subsample));
        save(filename, ...
            'acc','acc1','acc2','acc3','cmatrix','cmatrix_norm','predict_labels','cov_prior_coeff','time','loglikelihood','loglikelihood_all','loglikelihood_all_cv','O','split_sub','max_iter_em','hyperparams_set','hyperparams_trace','params_set','dura_type','LLtrace','llhptrace_verbose','llhptrace','llhhtrace','-v7.3')%,'energy','w'
        if MC > 0 % if MC is large, saving 'params_samples_all_burnin' will need a lot of space
            if save_burnin
                save(filename,'params_samples_all','params_samples_all_burnin','llh_train','-append');
            end
        end
        
        if eval_train == 1
            save(filename,'acc_train','cmatrix_train','loglikelihood_train','loglikelihood_all_train','loglikelihood_all_train_cv','cmatrix_norm_train','-append');
        end
        
        if save_for_inference % currently only save the last fold of 'split'
            %%
            for a = 1:num_class
                % save data and learned hyperparameters for mean field variational inference
                alpha(a) = hyperparams_set(a,cv).hyperparams.trans_sym;
                alpha_0(:,a) = hyperparams_set(a,cv).hyperparams.dura(:,1); % must be Poisson duration
                beta_0(:,a) = hyperparams_set(a,cv).hyperparams.dura(:,2);
                mu_0(:,a) = hyperparams_set(a,cv).hyperparams.emis.mu;
                sigma_0(:,:,a) = hyperparams_set(a,cv).hyperparams.emis.S;
                kappa_0(a) = hyperparams_set(a,cv).hyperparams.emis.kappa;  %%%%%%%%% right now is not estimated
                nu_0(a) = hyperparams_set(a,cv).hyperparams.emis.nu; %%%%%%%%% right now is not estimated
            end
            save(fullfile(dir_base,dataset_name{dataset_id},strcat('dataset_train_',dataset_name{dataset_id},sprintf('_maxiter%d',max_iter))), ...
                'dataset_train','O','Q','L','alpha','alpha_0','beta_0','mu_0','sigma_0','kappa_0','nu_0');
            %% initial guess of hyperparameters            
            for a = 1:num_class
                % save data and learned hyperparameters for mean field variational inference
                alpha(a) = dirichlet_sym_mle(mk_stochastic(hyperparams_trace{a,cv}(1).hyperparams.trans),0);
                alpha_0(:,a) = hyperparams_trace{a,cv}(1).hyperparams.dura(:,1); % must be Poisson duration
                beta_0(:,a) = hyperparams_trace{a,cv}(1).hyperparams.dura(:,2);
                mu_0(:,a) = hyperparams_trace{a,cv}(1).hyperparams.emis.mu;
                sigma_0(:,:,a) = hyperparams_trace{a,cv}(1).hyperparams.emis.S;
                kappa_0(a) = hyperparams_trace{a,cv}(1).hyperparams.emis.kappa;  %%%%%%%%% right now is not estimated
                nu_0(a) = hyperparams_trace{a,cv}(1).hyperparams.emis.nu; %%%%%%%%% right now is not estimated
            end
            save(fullfile(dir_base,dataset_name{dataset_id},strcat('dataset_train_',dataset_name{dataset_id},'_ini')), ...
                'dataset_train','O','Q','L','alpha','alpha_0','beta_0','mu_0','sigma_0','kappa_0','nu_0');
        end
        fprintf('State %d completed\n\n',Q(it));
    end
    fprintf('Fold %d completed\n\n\n',cv);
end