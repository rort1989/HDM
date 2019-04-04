function [x,flag,cs,As,bs,xs,lam,s,hist,optval,time] = iplp(Problem,tol,maxit,varargin)
% IPLP Interior Point LP Solver
% Author: Rui Zhao
% Date: 2012.05.01
% 
% x = iplp(Problem) solve the linear program:
%
%   minimize c'*x where Ax = b and lo <= x <= hi
%
% where the variables are stored in the following struct:
%
%   Problem.A
%   Problem.aux.c
%   Problem.aux.lo
%   Problem.aux.hi
%   Problem.b
%  
% This function will neither work for unbounded constraints (i.e.
% lo=-Inf, hi=Inf) nor for infeasible problem (i.e. Ax=b is an incosistent
% system).
%
% [...] = iplp(Problem,tol,maxit) solves the problem to a point where
% the duality measure (xs'*s)/n <= tol and the normalized residual
%  norm([As'*lam + s - cs; As*xs - bs; xs.*s])./norm([bs;cs]) <= tol
% and fails if this takes more than maxit iterations.
%
% [x,flag,cs,As,bs,xs,lam,s,hist,optval,time] = iplp(Problem)
% also returns a flag indicating whether or not the
% solution succeeded (flag = 0 => success and flag = 1 => failure),
% along with the solution for the problem converted to standard form (xs):
%   minimize cs'*xs where As*xs = bs and xs >= 0
% and the associated Lagrange multipliers (lam, s).
% hist is a two by i matrix, where i is the number of iterations when the
% program terminates. The first row of hist records the values of
% normalized residual. The second row has the values of duality measure at
% each iteration.
% optval is the optimal value of the object function when the program
% terminates.
% time is an estimate cost of the entire running time of the program
% excluding the time of loading data.
%
% The option is:
%
%   'decmp' : decide the function used to compute the matrix factorization
%             when solving the linear system at each iteration
%
% The values of the option
%
%   'cholinc' : use function cholinc to compute matrix factorization,
%   default method if the option is omitted
%   'ldl'     : use function ldl to compute matrix factorization
%
% Example: 
%   x = iplp('lp_fit1d.mat',1e-9,50,'decmp','ldl');


data = load(Problem);
A = data.Problem.A;
b = data.Problem.b;
c = data.Problem.aux.c;
lo= data.Problem.aux.lo;
hi= data.Problem.aux.hi;
start = cputime;

p = inputParser;
defaultDecmp = 'cholinc';
validDecmp = {'cholinc','ldl'};
checkDecmp = @(x) any(validatestring(x,validDecmp));

addOptional(p,'decmp',defaultDecmp,checkDecmp);

p.parse(varargin{:});
opt = p.Results;

% make sure the dimension of A,b,c are correct
[m,n] = size(A);
if m~=length(b) || n~=length(c)
    error('size of A,b,c does not match');
end
% dealing with A rank deficient case
r=sprank(A);
if  r < m
    [Qa Ra] = qr(A);
    A = Ra(1:r,:);
    b = Qa'*b;
    b = b(1:r);
    m = r;
end
[m0 n0] = size(A);
% add slack variables to form new linear system that make the feasible
% region becomes entire non-negative numbers
if ~isempty(find(lo>0,1))
    ind_lo = find(lo>0);
    lo_len = length(ind_lo);
    % fast way to form the new system without using for loop
    Al = zeros(m+lo_len,n+lo_len);
    Al(1:m,1:n) = A;
    Al(m+1:end,n+1:end) = -eye(lo_len);
    dgl = eye(n);
    Al(m+1:end,1:n) = dgl(ind_lo,:);
    b = [b; lo(ind_lo)];
    c = [c; zeros(lo_len,1)];% make sure the slack variables does not affect the object function value    
    A = Al;
end
if ~isempty(find(hi<Inf,1))
    ind_hi = find(hi<Inf);
    hi_len = length(ind_hi);
    [mAl nAl] = size(A);
    % fast way to form the new system without using for loop
    Ah = zeros(mAl+hi_len,nAl+hi_len);
    Ah(1:mAl,1:nAl) = A;
    Ah(mAl+1:end,nAl+1:end) = eye(hi_len);
    dgh = eye(nAl);
    Ah(mAl+1:end,1:nAl) = dgh(ind_hi,:);
    b = [b; hi(ind_hi)];
    c = [c; zeros(hi_len,1)];    
    A = Ah;
end
[m n] = size(A);
lo = zeros(n,1);
hi = Inf*ones(n,1);
[A0,b0,x0,lambda0,s0] = startpoint(A,b,c,m,n);
A = A0;
b = b0;
[m n] = size(A);
[xs optval s lambda exitflag hist] = SPF(A,b,c,lo,hi,x0,lambda0,s0,maxit,m,n,tol,opt.decmp);
% assign output
time = cputime - start;
x = xs(1:n0);
s = s(1:n0);
lam = lambda(1:m0);
As = A;
bs = b;
cs = c;
flag = exitflag;

function [A0,b0,x0,lambda0,s0] = startpoint(A,b,c,m,n)
    % taking care of the case that A is rank deficient
    rA = sprank(A);
    if rA < m
        warning('A is rank deficient. Gaussian elimination will be performed on A to eliminate the dependent row');
        [Q R] = qr(A);
        Abar = Q'*A;
        bbar = Q'*b;
        Abar = Abar(1:rA,:);
        bbar = bbar(1:rA);
        A = Abar;
        b = bbar;    
    end
    mat = (A*A');
    x_tilda = A'/mat*b;
    lambda_tilda = mat\A*c;
    [lastmsg,lastid] = lastwarn;
    if ~isempty(lastid)        
        warning('off',lastid);
    end
    s_tilda = c - A'*lambda_tilda;
    % make sure x,s are positive
    delta_x = max(-1.5*min(x_tilda),0);
    delta_s = max(-1.5*min(s_tilda),0);
    x_hat = x_tilda + delta_x*ones(n,1);
    s_hat = s_tilda + delta_s*ones(n,1);
    % avoid too close to zero
    nomi = x_hat'*s_hat;
    if sum(s_hat) == 0 || sum(x_hat) == 0
        s0 = s_hat + delta_s*ones(n,1);
        x0 = x_hat + delta_x*ones(n,1);
    else
        delta_x_hat = 0.5*nomi/sum(s_hat);        
        delta_s_hat = 0.5*nomi/sum(x_hat);
        x0 = x_hat + delta_x_hat*ones(n,1);
        s0 = s_hat + delta_s_hat*ones(n,1);
    end    
    lambda0 = lambda_tilda;
    A0 = A;
    b0 = b;
end

% Primal-Dual Path-Following
function [x optval s lambda exitflag hist] = SPF(A,b,c,lo,hi,x0,lambda0,s0,maxiter,m,n,tol,decmp)
    x = x0;
    lambda = lambda0;
    s = s0;
    exitflag = 1;
    h = zeros(2,maxiter);
    for k = 1:maxiter
        % w/o taking infeasible point into consideration
        % step 1 solve predict system
        % define the following residuals
        rb = A*x - b;
        rc = A'*lambda + s - c;
        d = s.^(-1).*x;   
        D = diag(d.^(0.5));
        B = A*(D.^2)*A';
        ss = size(B);
        rankB = sprank(B);
        ss(1);
        if rankB < ss(1)
            error('A*D^2*A^T is singular');
        end
        % only need to do factorization once
        if strcmp(decmp,'ldl')
            [L,DD] = ldl(B);
        else
            R = cholinc(sparse(B),'inf');
    	    [lastmsg,lastid] = lastwarn;
            if ~isempty(lastid)        
                warning('off',lastid);
            end
        end        
        
        % step 7 check optimality termination condition
        cond1 = norm([rb;rc;x.*s])/norm([b;c]); h(1,k) = cond1;
        cond2 = x'*s/n;   h(2,k) = cond2;
        if (cond1<=tol) && (cond2<=tol)   
            % converge to optimal value
            exitflag = 0;
            break;
        end
        rxs= x.*s;
        mu = x'*s/n;
        if strcmp(decmp,'ldl')
            [dx_aff dlambda_aff ds_aff] = linesys(A,DD,L,d,s,rb,rc,rxs);
        else
            [dx_aff dlambda_aff ds_aff] = linesys_new(A,R,d,s,rb,rc,rxs);
        end
        % step 2 calculate predict step and mu
        eta = 1;
        [alpha_pri_aff alpha_dua_aff] = steplength(dx_aff,ds_aff,x,s,eta);
        % step 3 set centering parameter sigma
        mu_aff = (x+alpha_pri_aff.*dx_aff)'*(s+alpha_dua_aff.*ds_aff)/n;
        sigma = (mu_aff/mu)^3;
        % step 4 solve correction system
        rxs_cor = rxs + dx_aff.*ds_aff - sigma.*mu.*ones(n,1);
        if strcmp(decmp,'ldl')
            [dx dlambda ds] = linesys(A,DD,L,d,s,rb,rc,rxs_cor);
        else
            [dx dlambda ds] = linesys_new(A,R,d,s,rb,rc,rxs_cor);
        end
        % step 5 calculate correction step
        eta = 1 - exp(-k/10-1.4);
        %%%%%%%this is an interesting part, if eta get close to 1 too fast, may cause problem
        [alpha_pri alpha_dua] = steplength(dx,ds,x,s,eta);
        % step 6 update x,lambda,s
        x = x + alpha_pri.*dx;
        lambda = lambda + alpha_dua.*dlambda;
        s = s + alpha_dua.*ds;
        
    end
    if sum(x<lo) || sum(x>hi)
        error('infeasible solution');
    end
    hist = h(:,1:k);
    optval = c'*x;    
end

% solve linear system in each iteration using LDL
function [dx dlambda ds] = linesys(A,DD,L,d,s,rb,rc,rxs)
    sxs = s.^(-1).*rxs;
    rhs_lambda = -rb - A*(d.*rc) + A*sxs;
    z = L\rhs_lambda;
    [lastmsg,lastid] = lastwarn;
    if ~isempty(lastid)        
        warning('off',lastid);
    end
    y = DD\z;
    if ~isempty(lastid)        
        warning('off',lastid);
    end
    dlambda = L'\y;
    if ~isempty(lastid)        
        warning('off',lastid);
    end
    ds = -rc - A'*dlambda;
    dx = -sxs - d.*ds;
end

% solve linear system in each iteration using cholinc
function [dx dlambda ds] = linesys_new(A,R,d,s,rb,rc,rxs)
    sxs = s.^(-1).*rxs;
    rhs_lambda = -rb - A*(d.*rc) + A*sxs;
    y = (R')\rhs_lambda;
    [lastmsg,lastid] = lastwarn;
    if ~isempty(lastid)        
        warning('off',lastid);
    end
    dlambda = R\y;
    if ~isempty(lastid)        
        warning('off',lastid);
    end
    ds = -rc - A'*dlambda;
    dx = -sxs - d.*ds;
end

% compute step length of updating
function [alpha_pri alpha_dua] = steplength(dx,ds,x,s,eta)
    ind_pri = find(dx<0); %
    ind_dua = find(ds<0); % must not include equality
    if isempty(ind_pri)
        alpha_pri = 1;
    else
        alpha_pri = min(-x(ind_pri)./dx(ind_pri));
        alpha_pri = min(1,eta*alpha_pri);
    end
    if isempty(ind_dua)
        alpha_dua = 1;
    else        
        alpha_dua = min(-s(ind_dua)./ds(ind_dua));        
        alpha_dua = min(1,eta*alpha_dua);
    end       
end

end