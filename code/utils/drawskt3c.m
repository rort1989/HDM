%USAGE: drawskt3(data) --- show actions specified by data
% draw with specified color on each data point
function drawskt3c(X,Y,Z,C,joints,topology,framerate,displayrange,holdon,cc)

% data: total number of joints*3
% topology: total number of joints*1 specifing parent node of each node

if nargin < 7
    framerate = 1/20;
end
if nargin < 8
    xmax = max(max(X)); xmin = min(min(X)); 
    ymax = max(max(Y)); ymin = min(min(Y));
    zmax = max(max(Z)); zmin = min(min(Z));
    displayrange = [xmin xmax ymin ymax zmin zmax];
end
if nargin < 9
    holdon = 0;
end
if nargin < 10
    [colors,~,C] = unique(C);
    num_color = length(colors);
    cc = hsv(num_color);
end

J = size(X,1);
for s=1:size(X,2)
    S=[X(:,s) Y(:,s) Z(:,s)];
    S_max = max(S);
    S_min = min(S);
  
    xlim = [0 800];
    ylim = [0 800];
    zlim = [0 800];
    set(gca, 'xlim', xlim, ...
             'ylim', ylim, ...
             'zlim', zlim);
    if ~holdon
        h=plot3(S(:,1),S(:,2),S(:,3),'.'); grid on
    else
        h=plot3(S(:,1),S(:,2),S(:,3),'.','color',cc(C(s),:)); grid on; hold on;
    end
    %rotate(h,[0 45], -180);
    set(gca,'DataAspectRatio',[1 1 1]) 
    title(num2str(s));
    axis(displayrange) % corresponds to xzy in right-hand camera coord
    for j=1:J        
        child = joints(j);        
        par = topology(child);
        if par == 0
            continue;
        end
        i = find(joints==par);
        line([S(j,1) S(i,1)], [S(j,2) S(i,2)], [S(j,3) S(i,3)],'color',cc(C(s),:));
    end
    pause(framerate)
end
