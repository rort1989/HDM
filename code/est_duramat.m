function [duramat, dura_truncated, dura] = est_duramat(IDX,Q,L)

T = length(IDX);
duramat = zeros(Q,L);
dura = zeros(Q,T);
t = 1;
count = 1;
while t < T
    q = IDX(t);
    if q == IDX(t+1);
        count = count + 1;        
    else
        dura(q,count) = dura(q,count) + 1;
        count = 1;
    end
    if t+1 == T
        dura(IDX(T),count) = dura(IDX(T),count) + 1;
    end
    t = t + 1;    
end
if T > L
    duramat(:,1:L-1) = dura(:,1:L-1);
    duramat(:,L) = sum(dura(:,L:end),2); % duration equal to or greater than L are aggregated together
else
    duramat(:,1:T) = dura;
end
dura_truncated = duramat; % truncated raw count
duramat = mk_stochastic(duramat);
