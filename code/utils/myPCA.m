% PCA
function [w, eigvalue, proj, mean_train, var_train] = myPCA(data, D_low, bias)

if nargin < 3
    bias = 0;
end

Numcases = size(data,2);
mean_train = zeros(size(data,1),1);
if ~bias
mean_train = mean(data,2);
data = bsxfun(@minus, data, mean(data,2));
end

% the inputData is mean-subtracted
C = 1/(Numcases-1).*data*data';
var_train = diag(C);

% PCA
[u,val] = eig(C);
diagval = diag(val);
% [val,u] = eigdec(C,size(data,1));
% diagval = val;
[~, rindices] = sort(-1.*diagval);
eigvalue = diagval(rindices);
u = u(:,rindices);
w = u(:,1:D_low);
proj = w'*data;