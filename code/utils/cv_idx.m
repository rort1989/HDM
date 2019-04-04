% create index for K-fold cross validation
function idx = cv_idx(N,K)
idx = struct('train',cell(1,K),'validation',cell(1,K));
if K >= 2
step = floor(N/K);
for k = 1:K-1
    validation_start = (k-1)*step + 1;
    validation_end = k*step;
    validation_idx = validation_start:validation_end;
    train_idx = setdiff(1:N,validation_idx);
    idx(k).validation = validation_idx;
    idx(k).train = train_idx;
end
idx(K).validation = (K-1)*step+1:N;
idx(K).train = 1:(K-1)*step;
else % validation equals training
idx.validation = 1:N;
idx.train = 1:N;
end