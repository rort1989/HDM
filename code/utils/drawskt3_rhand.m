%USAGE: drawskt3(data) --- show actions specified by data
function drawskt3_rhand(X,Y,Z,joints,topology,framerate,displayrange,holdon,name,varargin)

% data: total number of joints*3
% topology: total number of joints*1 specifing parent node of each node

if nargin < 6
    framerate = 1/20;
end
if nargin < 7
    xmax = max(max(X)); xmin = min(min(X)); 
    ymax = max(max(Y)); ymin = min(min(Y));
    zmax = max(max(Z)); zmin = min(min(Z));
    displayrange = [xmin xmax ymin ymax zmin zmax];
end
if nargin <8
    holdon = 0;
end
title_flag = 1;
if nargin < 9
    name = [];
    title_flag = 0;
end

p = inputParser;
default_MarkerSize = 5;
default_LineWidth = 1;
addOptional(p,'MarkerSize',default_MarkerSize,@isnumeric);
addOptional(p,'LineWidth',default_LineWidth,@isnumeric);
p.parse(varargin{:});
MarkerSize = p.Results.MarkerSize;
LineWidth = p.Results.LineWidth;

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
        h=plot3(S(:,1),S(:,2),S(:,3),'r.','MarkerSize',MarkerSize); grid on
    else
        h=plot3(S(:,1),S(:,2),S(:,3),'r.','MarkerSize',MarkerSize); grid on; hold on;
    end
    %rotate(h,[0 45], -180);
    set(gca,'DataAspectRatio',[1 1 1])
    if title_flag == 0
        title(num2str(s));
    else
        title(name);
    end    
    axis(displayrange) % corresponds to xzy in right-hand camera coord
    for j=1:J        
        child = joints(j);        
        par = topology(child);
        if par == 0
            continue;
        end
        i = find(joints==par);
        if par == 9 || par == 10 || par == 11 || par == 12 % right shoulder,elbow,wrist,hand
            line([S(j,1) S(i,1)], [S(j,2) S(i,2)], [S(j,3) S(i,3)],'Color','g','LineWidth',LineWidth);
        else
            line([S(j,1) S(i,1)], [S(j,2) S(i,2)], [S(j,3) S(i,3)],'LineWidth',LineWidth);
        end
    end
    pause(framerate)
end
