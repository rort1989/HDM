function [datacells, params] = getParams(data_struct, stateSeq, dist_struct, theta, assignment, discrete, HMM_type) %, cov_type
% extract parameters from current sample, convert to BNT format
N = length(data_struct); % number of sequences
datacells = cell(1,N);
stateZ = [];
%stateS = [];
for n = 1:N
    datacells{n} = data_struct(n).obs;
    if strcmp(HMM_type,'HDP') % 1) use actual sampled hidden state
        stateZ = union(stateZ, unique(stateSeq(n).z));
        %stateS = union(stateS, unique(stateSeq(n).s));
    end
end
if strcmp(HMM_type,'finite') % 2) use max number of hidden state;
    stateZ = 1:length(dist_struct.pi_init);    
end
stateS = 1:size(dist_struct.pi_s,2); % current did not restrict substate
% find a permutation on stateS
% if length(stateS) > 1
%     sigma = var(squeeze(theta.mu(:,stateZ(1),:)),0,2);
%     [~,idx] = max(sigma);
%     [~,perm] = sort(squeeze(theta.mu(idx,stateZ(1),:)));
%     stateS = perm;
% end
params.prior = dist_struct.pi_init(stateZ); params.prior = params.prior(:);
params.transmat = dist_struct.pi_z(stateZ,stateZ);
if ~discrete
    params.mu = theta.mu(:,stateZ,stateS);
    O = size(theta.invSigma,1);
    params.sigma = zeros(O,O,length(stateZ),length(stateS));
    invCov = theta.invSigma(:,:,stateZ,stateS);
    for s = 1:length(stateS)
        for z = 1:length(stateZ)
            params.sigma(:,:,z,s) = inv(invCov(:,:,z,s));
        end
    end
    params.mixmat = dist_struct.pi_s(stateZ,stateS);
else
    params.obsmat = squeeze(theta.p);
end

if strcmp(HMM_type,'HDP') % need re-normalization
    params.prior = params.prior/sum(params.prior);
    params.transmat = bsxfun(@rdivide,params.transmat,sum(params.transmat,2));
    if ~discrete
        params.mixmat = bsxfun(@rdivide,params.mixmat,sum(params.mixmat,2));
    else
        params.obsmat = bsxfun(@rdivide,params.obsmat,sum(params.obsmat,2));
    end
end
params.assignment = assignment;

end