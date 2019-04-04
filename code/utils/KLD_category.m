function [D1, D2] = KLD_category(alpha, beta)
% [D1, D2] = category_KLD(alpha, beta)
% compute KL-divergence between two category distribution given their
% respective parameters alpha and beta which are two vectors of same length
% D1 = KL(p(alpha)||q(beta))
% D2 = KL(q(beta)||p(alpha))

if length(alpha) ~= length(beta)
    error('two input must have the same length')
end
alpha = alpha(:); 
alpha(alpha==0) = eps;
beta = beta(:); 
beta(beta==0) = eps;
D1 = alpha'*(log(alpha) - log(beta));
D2 = beta'*(log(beta) - log(alpha));