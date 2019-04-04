%USAGE: drawskt3(data) --- show actions specified by data
function drawskt3(X,Y,Z,joints,topology,varargin)

% data: total number of joints*3
% topology: total number of joints*1 specifing parent node of each node
% for Kinect V1: topology = [0 1 2 3 3 5 6 7 3 9 10 11 1 13 14 15 1 17 18 19];
% for Kinect V2: topology = [0 1 21 3 21 5 6 7 21 9 10 11 1 13 14 15 1 17 18 19 2 8 7 12 11];

% load optional input
p = inputParser;
default_framerate = 0.01;
default_holdon = 0;
default_MarkerSize = 5;
default_LineWidth = 1;
default_jointID = false;
xmax = max(max(X)); xmin = min(min(X)); 
ymax = max(max(Y)); ymin = min(min(Y));
zmax = max(max(Z)); zmin = min(min(Z));
default_displayrange = [xmin xmax ymin ymax zmin zmax];
addOptional(p,'displayrange',default_displayrange,@isnumeric);
addOptional(p,'framerate',default_framerate,@isnumeric);
addOptional(p,'holdon',default_holdon,@isnumeric);
addOptional(p,'MarkerSize',default_MarkerSize,@isnumeric);
addOptional(p,'LineWidth',default_LineWidth,@isnumeric);
addOptional(p,'jointID',default_jointID,@islogical);
p.parse(varargin{:});
displayrange = p.Results.displayrange;
framerate = p.Results.framerate;
holdon = p.Results.holdon;
MarkerSize = p.Results.MarkerSize;
LineWidth = p.Results.LineWidth;
jointID = p.Results.jointID;

J = length(joints);

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
    title(num2str(s));
    axis(displayrange) % corresponds to xzy in right-hand camera coord
    for j=1:J
        child = joints(j);        
        par = topology(child);
        if jointID
            text(S(j,1),S(j,2),S(j,3),num2str(j))
        end
        if par == 0
            continue;
        end
        i = find(joints==par);
        line([S(j,1) S(i,1)], [S(j,2) S(i,2)], [S(j,3) S(i,3)],'LineWidth',LineWidth);        
    end
    pause(framerate)
end
