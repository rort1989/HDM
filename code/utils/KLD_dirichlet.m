function [D1, D2] = dirichlet_KLD(alpha, beta)
% compute KL-divergence between two Dirichlet distribution given their
% respective parameters alpha and beta which are two vectors of same length
% D1 = KL(p(alpha)||q(beta))
% D2 = KL(q(beta)||p(alpha))
% Reference: Baris Kurt 2013
% http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/#comment-29514
if length(alpha) ~= length(beta)
    error('two input must have the same length')
end
alpha = alpha(:);
beta = beta(:);
cnst = gammaln(sum(alpha)) - gammaln(sum(beta)) - sum(gammaln(alpha)) + sum(gammaln(beta));
D1 = cnst + (alpha - beta)' * (psi(alpha) - psi(sum(alpha)));
D2 = -cnst + (beta - alpha)' * (psi(beta) - psi(sum(beta)));