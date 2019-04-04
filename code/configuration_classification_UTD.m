% define constant for classification experiment
config = 1;% 0. raw selected joint positions; 1. PCA Location+Motion; 2. PCA whitening Location+Motion
use_motion = 1; % whether to include motion features
if use_motion == 0
    clear feature2_
end
d_selected = []; % joint positions selected, if empty, select all
idx_exclude = []; % outlier sequence if any for training
subsample = 1;
FB = 0; % whether to use full Bayesian inference
MC = 10 + 20*(FB==1);
burnin = 20;
eval_train = 1; % whether to evaluate training accuracy
test = 1; % different testing scheme e.g. 1. cross subject, cross view
portion_missing_train = 0; % portion of training data that is missing, data amount = D*T
portion_missing_test = 0; % portion of testing data that is missing, data amount = D*T

gestures = 1:27;
% cross subjects
if test == 1
    split_sub.train = 1:2:7;
    split_sub.validation = 2:2:8;
else
    subjects = 1:length(unique(labels(:,2)));
    split_sub = cv_idx(length(subjects),test);
end

num_class = length(gestures);
num_inst = size(labels,1);
Q = 12;
M = 1;
stride = 1;
portion = 0.99;
max_iter = 10;
max_iter_em = 10; 
useprior = 2; % 0. no hyperparams; 1. fixed hyperparams; 2. learned hyperparams
adapt = 0; % 0. fixed params for all sequences, 1. adapt params for individual sequence
cov_prior_coeff_portion = 0.2; % >0 HHSMM+spatial prior
cov_prior_coeff = 0.01;
dura_prior_coeff = 1; % USE 1/K heuristics
cov_type = 'full'; %%%%%%%% need to modify to support diagonal covariance case
dura_type = 'Poisson'; % either 'Multinomial' or 'Poisson'
topology = [0 1 2 3 3 5 6 7 3 9 10 11 1 13 14 15 1 17 18 19]; % Kinect 1 standard topology
njoints = length(topology);