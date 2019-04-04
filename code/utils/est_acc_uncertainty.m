function [total_uncertainty_sort,idx,accuracy_model_avg_portion,accuracy_model_avg_portion_within,labels_model_avg_sort,true_labels_sort]= est_acc_uncertainty(predict_labels,true_labels,total_uncertainty,portion_uncertainty)

P = length(portion_uncertainty);
[total_uncertainty_sort, idx] = sort(total_uncertainty,'ascend');
labels_model_avg_sort = predict_labels(idx);
true_labels_sort = true_labels(idx);

accuracy_model_avg_portion = zeros(P,1);
accuracy_model_avg_portion_within = zeros(P,1);
labels_model_avg_sort_portions = cell(P,1);
true_labels_sort_portions = cell(P,1);
for p = 1:P
    idx_end = portion_uncertainty(p)*length(true_labels_sort)/100;
    accuracy_model_avg_portion(p) = sum(labels_model_avg_sort(1:round(idx_end))==true_labels_sort(1:round(idx_end)))/round(idx_end);
    labels_model_avg_sort_portions{p} = labels_model_avg_sort(1:round(idx_end));
    true_labels_sort_portions{p} = true_labels_sort(1:round(idx_end));
    % accuracy within each category (different from accumulate accuracy)
    if p == 1
        idx_start = 1;
    else
        idx_start = portion_uncertainty(p-1)*length(true_labels_sort)/100+1;
    end
    accuracy_model_avg_portion_within(p) = sum(labels_model_avg_sort(round(idx_start):round(idx_end))==true_labels_sort(round(idx_start):round(idx_end)))/(round(idx_end)-round(idx_start)+1);
end