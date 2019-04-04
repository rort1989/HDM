% define constant for classification experiment
config = 0;% 0. raw selected joint positions; 1. PCA Location+Motion; 2. PCA whitening Location+Motion
use_motion = 0; % whether to include motion features
if use_motion == 0
    clear feature2_
end
d_selected = [1 2 7 8 13 14 19 20 25 26]; % joint positions selected, if empty, select all
idx_exclude = [711 712 753 1058 1146 961 965 972 976 977 980 993 1307 1635]; % outlier sequence if any for training
subsample = 1;
FB = 0; % whether to use full Bayesian inference
MC = 10 + 20*(FB==1);
burnin = 20;
eval_train = 1; % whether to evaluate training accuracy
test = 1;

gestures = setdiff(1:length(unique(labels(:,1))),[4 7 9 15]); % number of gestures
% subjects = 1:length(unique(labels(:,2)));
% split_sub = cv_idx(length(subjects),length(subjects));
% cross subjects
split_sub.train = 1;
split_sub.validation = 2;

num_class = length(gestures);
num_inst = size(labels,1);
Q = 15;
M = 1;
stride = 1;
portion = 0.9;
max_iter = 10;%10; 
max_iter_em = 20;
useprior = 2; % 0. no hyperparams; 1. fixed hyperparams; 2. learned hyperparams
adapt = 0; % 0. fixed params for all sequences, 1. adapt params for individual sequence
cov_prior_coeff_portion = 0.2; % >0 HHSMM+spatial prior
cov_prior_coeff = 100;
dura_prior_coeff = 0.01; % USE 1/K heuristics
cov_type = 'full'; %%%%%%%% need to modify to support diagonal covariance case
dura_type = 'Poisson'; %'Multinomial';% % either 'Multinomial' or 'Poisson'