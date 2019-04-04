function [cm_raw, cm_norm] = cm(class_num,test_label,predict_label)
% compute confusion matrix (without and with normalization with each column)
% [cm_raw, cm_norm] = cm(class_num,test_label,predict_label)
% label values in test_label and predict_label must be ranging from 1 to
% class_num

N = class_num;
cm_raw = zeros(N,N);
for i = 1:length(predict_label)
    cm_raw(test_label(i),predict_label(i)) = cm_raw(test_label(i),predict_label(i)) + 1;
end
cm_norm = bsxfun(@rdivide,cm_raw,sum(cm_raw,2));