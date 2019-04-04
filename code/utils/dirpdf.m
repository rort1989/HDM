function y = dirpdf(x,a)
%DIRPDF Dirichlet probability density function (pdf).
%   Y = DIRPDF(X,ALPHA) returns the pdf for the Dirichlet distribution with
%   parameters ALPHA, evaluated at each row of X. X can be a N-by-D or
%   N-by-(D+1) matrix or a 1-by-D or 1-by-(D+1) vector, where D is the 
%   number of random variables. ALPHA is a N-by-(D+1) matrix or a 1-by-(D+1)
%   vector. Each row of X must be less than or equal to one. If a vector of
%   D+1 is given the sum of X must equal 1. Y is a N-by-1 vector, and DIRPDF 
%   computes each row of Y using the corresponding rows of the inputs.
%
%   Example:
%    Calculate the Dirichlet probability density distribution at X using 
%    parameters ALPHA. 
%    ALPHA=[2,5,4];
%    X=[0.4,0.3,0.3];
%    Y=dirpdf(X,P);


%   See also DIRFIT, DIRRVAL, DIRRND, DIRCDF
%   Created on 27 Nov 09 by AndrewNOConnor@gmail.com


if nargin < 2
    error('dirpdf:TooFewInputs', ...
          'Requires two input arguments.');
end

%if a is a column vector, we transpose it.
if size(a,2)==1 && size(a,1) > 1
    a = a';
    if size(x,2)==1 && size(x,1) > 1  %transpose x if x is a column vector
        x = x';
    end
end

[n_a,d_a] = size(a);
[n_x,d_x] = size(x);

%Add extra column to x to make sum(x)=1
if d_a==d_x+1
    x(:,d_x+1)=1-sum(x,2);
    d_x=d_x+1;
end

if d_a~=d_x
    error('dirpdf:InputSizeMismatch', ...
          'Columns in X must be equal or one less than columns in ALPHA');
elseif n_a == 1 %when n_x>1
    a = repmat(a, n_x,1);
elseif n_x == 1
    x = repmat(x, n_a,1);
elseif n_a ~= n_x
    error('dirpdf:InputSizeMismatch', ...
          'X and ALPHA must have the same number of rows, or either can be a row vector.');        
end
d = d_a;
n = n_a;

%Exclude any rows where sum x doesn't equal 1
sum_x = sum(x,2);
Badx1 = sum_x < 1 - eps; %sum_x~=1;  %sum ~= 1
Badx2 = sum(x>1+eps|x<0-eps,2)>0; %row contains out of bound x
Badx  = Badx1|Badx2;
if sum(Badx)>0
    warning('dirpdf:ValuesOutOfBound',': X has values out of bound.');
end
x(Badx,:) = [ones(sum(Badx),1), zeros(sum(Badx),d-1)];


%Exclude any rows where a<0
Bada = sum(a<0,2)>0; %row contains out of bound a
if sum(Bada)>0
    warning('dirpdf:ValuesOutOfBound',': ALPHA has values out of bound.');
end
a(Bada,:) = ones(sum(Bada),d);


%Calculate Function
y = zeros(n,1); %Initialize output
t = ~(Bada | Badx); %Determine 'proper' rows
y(t) = prod(x(t,:).^(a(t,:)-1),2).*exp(gammaln(sum(a(t,:),2))-sum(gammaln(a(t,:)),2));

% Return NaN for invalid parameter values
Bad = Bada|Badx;
y(Bad) = NaN;
