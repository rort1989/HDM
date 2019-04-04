function [p,s] = fit_plane3D(pts)

% pts=[x,y,z] are n x 3 matrix of the three coordinates
% of a set of n points in three dimensions. s returns with
% the minimum mean square orthogonal distance to a least
% squares best-fit plane. The four coefficients of that
% plane's equation, Ax + By + Cz + D = 0, are returned in
% row vector p = [A,B,C,D]. A,B,C are normalized: A^2+B^2+C^2=1. 
% RAS - June 9, 2001

n = size(pts,1);
m = mean(pts);
w = bsxfun(@minus,pts,m); % Use "mean" point as base
a = (1/n)*(w')*w; % 'a' is a positive definite matrix
a(isnan(a)) = 0;
a(isinf(a)) = 0;
[u,d,v] = svd(a); % 'eig' & 'svd' get same eigenvalues for this matrix
p = u(:,3)'; % Get eigenvector for largest eigenvalue
p(4) = -p*m';
s = d(3,3); % Select the smallest eigenvalue
