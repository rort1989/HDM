function [path, delta, psi, tau] = viterbi_path_semi(prior, transmat, duramat, obslik)
% VITERBI Find the most-probable (Viterbi) path through the HSMM state trellis.
% explicit duration HSMM
% path = viterbi(prior, transmat, duramat, obslik)
%
% Inputs:
% prior(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
% obslik(i,t) = Pr(y(t) | Q(t)=i)
%
% Outputs:
% path(t) = q(t), where q1 ... qT is the argmax of the above expression.


% delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
% psi(j,t) = the best predecessor state, given that we ended up in state j at t
% tau(j,t) = the best duration of current state

% scaled = 1;

T = size(obslik, 2);
prior = prior(:);
Q = length(prior);
L = size(duramat,2);

delta = zeros(Q,T);
psi = zeros(Q,T);
tau = zeros(Q,T);
path = zeros(1,T);
% scale = ones(1,T);
loglik = log(obslik);
logtransmat = log(transmat);
logduramat = log(duramat);

t=1;
delta(:,t) = log(prior) + loglik(:,t);
psi(:,t) = 0; % arbitrary value, since there is no predecessor to t=1
tau(:,t) = 1;
% delta(:,t) = normalise(delta(:,t));

for t=2:T
    loglik_ = cumsum(loglik(:,t:-1:max(1,t-L+1)),2); % may need to perform in log domain
    for j=1:Q        
        Lm = min(t-1,L);
        delta_ =zeros(1,Lm);
        psi_ = zeros(1,Lm);
        %logduramat_ = log(mk_stochastic(duramat(j,1:Lm)));
        % logic: [val,idx] = max(delta(:,t-d) .* transmat(:,n) * duramat(n,d) * prod(obslik(n,t-d+1:t)) ); % Qx1 % duramat(n,d) 
        for d = 1:Lm    
            [delta_(d), psi_(d)] = max( delta(:,t-d) + logtransmat(:,j) );
            delta_(d) = delta_(d) + logduramat(j,d) + loglik_(j,d); % logduramat_(d)
        end
        [delta(j,t), tau(j,t)] = max(delta_);
        psi(j,t) = psi_(tau(j,t));
    end
%     delta(:,t) = normalise(delta(:,t));
end

% [p, path(T)] = max(delta(:,T));
% for t=T-1:-1:1
%     path(t) = psi(path(t+1),t+1);
% end

% backtracking
[~, path(T)] = max(delta(:,T));
dura = zeros(1,T);
dura(T) = tau(path(T),T);
count = 0;
for t=T-1:-1:2
    if dura(t+1) > 1
        path(t) = path(t+1);
        dura(t) = dura(t+1) - 1;
        count = count + 1;
    else
        path(t) = psi(path(t+count+1),t+count+1);
        dura(t) = tau(path(t),t);
        count = 0;
    end
end
[~,path(1)] = max(delta(:,1));

% If scaled==0, p = prob_path(best_path)
% If scaled==1, p = Pr(replace sum with max and proceed as in the scaled forwards algo)
% Both are different from p(data) as computed using the sum-product (forwards) algorithm

