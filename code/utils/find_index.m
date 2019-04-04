function idx = find_index(labels,gesture,subject,instance,pose,other)
% input labels should be a N*4 matrix where N is total number of instances
%
%       idx = find_index(labels,gesture,subject,instance,pose)
%
% pose is an optional input

tf = ismember(labels(:,1),gesture(:)); % locate actions
idx = find(tf==1);
tf = ismember(labels(:,2),subject(:)); % locate subjects
idx = intersect(idx,find(tf==1));
tf = ismember(labels(:,3),instance(:)); % locate instances
idx = intersect(idx,find(tf==1));
if nargin >= 5
    tf = ismember(labels(:,4),pose(:)); % locate poses
    idx = intersect(idx,find(tf==1));
end
if nargin >= 6
    tf = ismember(labels(:,5),other(:)); % locate others
    idx = intersect(idx,find(tf==1));
end