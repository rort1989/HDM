function [idx_train,idx_test] = find_idx(labels,id_train,id_test)
% labels are a set of ids each correspond to one data instance
% id_train are unique id values for training
% id_test are unque id values for testing
% output
% idx_train are indices of each id in id_train in labels
% idx_test are indices of each id in id_test in labels

idx_train = [];
idx_test = [];
for i = 1:length(id_train)
    idx_train = union(idx_train,find(labels==id_train(i)));
end
if nargin == 3
    for i = 1:length(id_test)
        idx_test = union(idx_test,find(labels==id_test(i)));
    end
end