function [path,delta,psi] = viterbi_path_HSMM(prior, transmat, duramat, obslik)
% VITERBI Find the most-probable (Viterbi) path through the HSMM state trellis.
% path = viterbi(prior, transmat, obslik)
%
% Inputs:
% prior(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
% duramat(i,d) = Pr(D(t)=d | Q(t)=i, D(t-1)=1)
% obslik(i,t) = Pr(y(t) | Q(t)=i)
%
% Outputs:
% path(t) = q(t), where q1 ... qT is the argmax of the above expression.

% delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
% psi(j,t) = the best predecessor state, given that we ended up in state j at t

scaled = 1;

T = size(obslik, 2);
prior = prior(:);
Q = length(prior);

delta = zeros(Q,T);
psi = zeros(Q,T);
path = zeros(1,T);
scale = ones(1,T);

t=1;
delta(:,t) = prior .* obslik(:,t);
if scaled
    [delta(:,t), norm_const] = normalise(delta(:,t));
    scale(t) = 1/norm_const;
end
psi(:,t) = 0; % arbitrary value, since there is no predecessor to t=1
for t=2:T
    for n=1:Q
        % find delta(n,t) and psi(n,t)
        max_value = -Inf;
        max_state = 0;
        for d = 1:t-1
%             duramat_ = mk_stochastic(duramat(n,1:t-1));
            % transmitted from a different state to j, where previous
            % state has last d time stamps
            [val,idx] = max(delta(:,t-d) .* transmat(:,n) * duramat(n,d) * prod(obslik(n,t-d+1:t)) ); % Qx1 % duramat(n,d) 
            if val > max_value
                max_value = val;
                max_state = idx;
            end
        end
        delta(n,t) = max_value ; %* obslik(n,t)
        psi(n,t) = max_state;
        %[delta(n,t), psi(n,t)] = max(delta(:,t-1) .* transmat(:,n));
    end
    if scaled
        [delta(:,t), norm_const] = normalise(delta(:,t));
        scale(t) = 1/norm_const;
    end
end
[p, path(T)] = max(delta(:,T));
for t=T-1:-1:1
    path(t) = psi(path(t+1),t+1);
end

% If scaled==0, p = prob_path(best_path)
% If scaled==1, p = Pr(replace sum with max and proceed as in the scaled forwards algo)
% Both are different from p(data) as computed using the sum-product (forwards) algorithm

if 0
if scaled
    loglik = -sum(log(scale));
    %loglik = prob_path(prior, transmat, obslik, path);
else
    loglik = log(p);
end
end
