function [dPQ, dQP] = KLD_Gaussian(mu1,sigma1,mu2,sigma2)
% Compute the KL-divergence of two univariate Gaussian from Q to P
% KL(P||Q) = \int P(x)\log(P(x)/Q(x))dx
% P = N(mu1,sigma1)
% Q = N(mu2,sigma2)

if length(mu1) == 1 % univariate case
    dPQ = 0.5*log(sigma2/sigma1) + ((mu1-mu2)^2+sigma1)/2/sigma2 - 0.5;
    dQP = 0.5*log(sigma1/sigma2) + ((mu1-mu2)^2+sigma2)/2/sigma1 - 0.5;
else % multivariate case
    dPQ = 0.5*(trace(sigma2\sigma1) + (mu1-mu2)'*(sigma2\(mu1-mu2)) - length(mu1) + log(det(sigma2)/det(sigma1)));
    dQP = 0.5*(trace(sigma1\sigma2) + (mu1-mu2)'*(sigma1\(mu1-mu2)) - length(mu1) + log(det(sigma1)/det(sigma2)));
end
