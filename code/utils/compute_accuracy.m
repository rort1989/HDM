function [acc_train,cmatrix_train,cmatrix_norm_train,predict_labels] = compute_accuracy(score,true_labels,num_class)
% assume score is N x num_class

[~,predict_labels] = max(score,[],2);
acc_train = sum(predict_labels(:)==true_labels(:))/length(predict_labels);
[cmatrix_train, cmatrix_norm_train] = cm(num_class,true_labels,predict_labels);
