%USAGE: drawskt3(data) --- show actions specified by data
function drawskt3cy(X,Y,Z,joints,topology,varargin)

% data: total number of joints*3
% topology: total number of joints*1 specifing parent node of each node

% load optional input
p = inputParser;
default_framerate = 0.01;
default_holdon = 0;
default_MarkerSize = 5;
default_FaceColor = [0 1 0];
default_LineWidth = 0.02;
xmax = max(max(X)); xmin = min(min(X)); 
ymax = max(max(Y)); ymin = min(min(Y));
zmax = max(max(Z)); zmin = min(min(Z));
default_displayrange = [xmin xmax ymin ymax zmin zmax];
addOptional(p,'displayrange',default_displayrange,@isnumeric);
addOptional(p,'framerate',default_framerate,@isnumeric);
addOptional(p,'holdon',default_holdon,@isnumeric);
addOptional(p,'MarkerSize',default_MarkerSize,@isnumeric);
addOptional(p,'FaceColor',default_FaceColor,@isnumeric);
addOptional(p,'LineWidth',default_LineWidth,@isnumeric);
p.parse(varargin{:});
displayrange = p.Results.displayrange;
framerate = p.Results.framerate;
holdon = p.Results.holdon;
MarkerSize = p.Results.MarkerSize;
FaceColor = p.Results.FaceColor;
LineWidth = p.Results.LineWidth;

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
	if ~isempty(FaceColor)
        if ~holdon
            h=plot3(S(:,1),S(:,2),S(:,3),'.','Color',FaceColor,'MarkerSize',MarkerSize); grid on
        else
            h=plot3(S(:,1),S(:,2),S(:,3),'.','Color',FaceColor,'MarkerSize',MarkerSize); grid on; hold on;
        end
    end
    for j=1:J        
        child = joints(j);        
        par = topology(child);
        %text(S(j,1),S(j,2),S(j,3),num2str(j))
        if par == 0
            continue;
        end
        i = find(joints==par);
        %line([S(j,1) S(i,1)], [S(j,2) S(i,2)], [S(j,3) S(i,3)],'LineWidth',LineWidth);        
        [CX, CY, CZ] = cylinder2P(LineWidth, 20,[S(joints(j),1) S(joints(j),2) S(joints(j),3)],[S(i,1) S(i,2) S(i,3)]); 
        if ~isempty(FaceColor)
            hold on;
            surf(CX,CY,CZ,'EdgeColor','none','LineStyle','none','FaceLighting','phong','FaceColor',FaceColor); 
        else
            surf(CX,CY,CZ,'EdgeColor','none','LineStyle','none','FaceLighting','phong'); hold on;
        end
    end
    set(gca,'DataAspectRatio',[1 1 1]) 
    %rotate(h,[0 45], -180);    
    title(num2str(s));
    axis(displayrange) % corresponds to xzy in right-hand camera coord
    hold off;
    pause(framerate)
end
