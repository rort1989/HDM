function [m_ess, V_ess] = hsmm_ess_emis(data, delta) %~  , ip
% compute expected sufficient statistics for mean, covariance
% Input:
%   delta is Q*T: delta(i,t) = P(Z(t) = i | X(1:T))
%   data is O*T
% Output:
%   m_ess(:,i) = \sum_{t=1}^T X(t)*P(Z(t)=i | X(1:T))
%   V_ess(:,:,i) = \sum_{t=1}^T X(t)*X(t)^T P(Z(t)=i | X(1:T))

[O,T] = size(data);
Q = size(delta,1);
V_ess = zeros(O,O,Q);
%~ debug for 'hsmm_mstep.m'
% ip = zeros(Q,1);

% m_ess = zeros(O,Q);
m_ess = data*delta'; % O*Q

% about 10 times slower for T ~= 100
% for t = 1:T
%     V = data(:,t)*data(:,t)';
%     for q = 1:Q
%         V_ess(:,:,q) = V_ess(:,:,q) + V*delta(q,t);
%     end
% end
for q = 1:Q
    wobs = data .* repmat(delta(q,:),O,1); % O*T
%     m_ess(:,q) = sum(wobs,2);
    V_ess(:,:,q) = wobs * data';
    %~ debug for 'hsmm_mstep.m'
    %ip(q) = sum(sum(wobs.*data,2));
end
