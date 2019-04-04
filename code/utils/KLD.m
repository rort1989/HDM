function [dPQ, dQP] = KLD(P,Q)
% Compute the KL-divergence of two distribution from Q to P
% KL(P||Q)
% P and Q are discrete number that corresponding to probability
% entries in P and Q sum to one
% P and Q are of same length

if length(P) ~= length(Q)
    error('Two probability vectors must have same length.')
end
if sum(P) ~= 1
    P = P/sum(P);
end
if sum(Q) ~= 1
    Q = Q/sum(Q);
end

% idx = find(P>0 & Q>0);
% dPQ = sum(P(idx).*(log(P(idx)) - log(Q(idx))));
% dQP = sum(Q(idx).*(log(Q(idx)) - log(P(idx))));
%%%%%%%%%%%%%% may not be correct
P(P==0) = eps;
Q(Q==0) = eps;
dPQ = sum(P.*(log(P) - log(Q)));
dQP = sum(Q.*(log(Q) - log(P)));
