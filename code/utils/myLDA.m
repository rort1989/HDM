% binary classification using LDA
function [acc, W] = myLDA(data_train, data_test, label_train, label_test, L)
% data_train is d*N_train matrix where d is dimension of each data sample and N
% is number of samples
% label_train is 1*N_train vector where ith entry gives the class label of ith
% data sample
% L is dimensionality of projected data, must be equal or less than d
[d_train, N_train] = size(data_train);
[d_test, N_test] = size(data_test);
tag = unique(label_train);
num_class = length(tag);
totalmean = mean(data_train,2);
if d_train ~= d_test
    error('training data and testing data must have same dimensionality');
else
    d = d_train;
    if L > d
        error('dimensionality of projected data, must be equal or less than original dimension');
    end
end
classmean=zeros(d,num_class);%matrix that store the value of each class mean
XB=zeros(d,num_class);
XW=zeros(d,N_train);
for i=1:num_class
    idx = label_train == tag(i);
    classmean(:,i)=mean(data_train(:,idx),2);
end
%divide SB=XB*XB';
for i=1:num_class
    XB(:,i)=classmean(:,i)-totalmean;%d*num_class
end
%divide SW=XW*XW';
for j=1:N_train
    XW(:,j)=data_train(:,j)-classmean(:,label_train(j));%d*N_train
end
[VB,valB]=eig(XB'*XB);
diagvalB=diag(valB);
[~, rindicesB]=sort(-1.*diagvalB);
diagvalB=diagvalB(rindicesB);
VB=VB(:,rindicesB);
V=XB*VB;
Y=V(:,1:L); % double check this point
DB=Y'*XB*XB'*Y;%directly compute XB*XB' will overflow memory
Z=Y*DB^(-0.5);
H=Z'*XW;
[U,valU]=eig(H*H');
diagvalU=diag(valU);
UU=U(:,1:L);
W=UU'*Z';

projected_data_train=zeros(L,N_train);
projected_data_test=zeros(L,N_test);
for i=1:N_train
    projected_data_train(:,i)=W*(data_train(:,i)-totalmean);
end
for i=1:N_test
   projected_data_test(:,i)=W*(data_test(:,i)-totalmean);
end
acc=test_LDA(projected_data_train,projected_data_test,label_train,label_test);

function accuracy=test_LDA(img_ref,img_test,label_train,label_test)
    % img_ref is N*d matrix, where N is number of data sample and d is
    % dim'l for each sample. Same size for img_test
    test_index=kdtreeidx(img_ref',img_test');
    predicted_labels = label_train(test_index);
    if length(predicted_labels) ~= length(label_test)
        error('predicted labels must have same length with true labels');
    end
    accuracy = sum(predicted_labels == label_test)/length(predicted_labels);
end

end