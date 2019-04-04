function S = mc_sample_dura(prior, trans, dura, len, numex, variant)
% SAMPLE_MC Generate random sequences from a semi-Markov chain.
% with duration on each state (should not have self-transition) follow a
% Gamma distribution

if nargin < 5
  numex = 1;
end
if nargin < 6
    variant = 1;
end

S = zeros(numex,len);
if variant == 1 % use Gamma distribution for duration
    for i=1:numex
        q = sample_discrete(prior);
        D = ceil(gamrnd(dura(q,1),1/dura(q,2)));
        t = min(D,len);
        S(i, 1:t) = q;
        while t < len
            tp = t;
            q = sample_discrete(trans(S(i,t),:));
            D = round(gamrnd(dura(q,1),1/dura(q,2)));
            t = min(t+D,len);
            S(i, tp+1:t) = q;
        end
    end
elseif variant == 2 % use Categorical distribution for duration
    for i=1:numex
        q = sample_discrete(prior);
        D = sample_discrete(dura(q,:));
        t = min(D,len);
        S(i, 1:t) = q;
        while t < len
            tp = t;
            q = sample_discrete(trans(S(i,t),:));
            D = sample_discrete(dura(q,:));
            t = min(t+D,len);
            S(i, tp+1:t) = q;
        end
    end
else % use Poisson distribution for duration
    for i=1:numex
        q = sample_discrete(prior);
        D = poissrnd(dura(q))+1;
        t = min(D,len);
        S(i, 1:t) = q;
        while t < len
            tp = t;
            q = sample_discrete(trans(S(i,t),:));
            D = round(poissrnd(dura(q)));
            t = min(t+D,len);
            S(i, tp+1:t) = q;
        end
    end
end
    