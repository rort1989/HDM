function [fs, acc, precision, recall, specificity] = fscore(tp,tn,fp,fn)
% accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned 
% precision = TP / (TP + FP) % for each class label 
% sensitivity = TP / (TP + FN) % for each class label 
% specificity = TN / (FP + TN) % for each class label 
% recall = sensitivity % for each class label 
% F-score = 2*TP /(2*TP + FP + FN) % for each class label 
% 
% TP: true positive, TN: true negative, 
% FP: false positive, FN: false negative
% p = tp(:) + fn(:);
% n = tn(:) + fp(:);
% idx_p = find(p~=0);
% idx_n = find(n~=0);
% if min(p(idx_p)) ~= 1 || max(p(idx_p))  ~=1 || min(n(idx_n)) ~= 1 || max(n(idx_n))  ~=1
%     error('probability does not sum up to 1');
% end
acc = (tp+tn)./(tp+tn+fp+fn);
precision = tp./(tp+fp);
recall = tp./(tp+fn); % same as sensitivity
specificity = tn./(fp+tn);
fs = 2*tp./(2*tp+fn+fp);