function [obs, hidden] = dhsmm_sample(initial_prob, transmat, duramat, obsmat, numex, len, variant)
% SAMPLE_DHMM Generate random sequences from a Hidden semi-Markov Model with discrete outputs.
% explicit duration on each state following gamma distribution
%
% [obs, hidden] = sample_dhmm(initial_prob, transmat, obsmat, numex, len)
% Each row of obs is an observation sequence of length len.

if nargin < 7
    variant = 1;
end

hidden = mc_sample_dura(initial_prob, transmat, duramat, len, numex, variant);
obs = multinomial_sample(hidden, obsmat);


