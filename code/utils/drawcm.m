function drawcm(cm, varargin)
% draw confusion matrix in gray scale color map
% normalization is performed within each row
% dependency: 'rotateXLabels.m'
%
% drawcm(cm, varargin)
% input:
%       cm: N*N confusion matrix
% optional argument
%       'Labels': an arrary of N cells corresponding to name of each class
%       'FontSize': real number corresponding to font size displayed in
%       confusion matrix entry
%       'Angle': rotation angle of x axis labels
%       'Colormap': colormap of figure
% example:
%       cm = rand(10)+5*eye(10);
%       Labels={'1','2','3','4','5','6','7','8','9','10'};
%       drawcm(cm,'FontSize',10,'Labels',Labels,'Angle',30);

[N, temp] = size(cm);
if N ~= temp
    error('Confusion matrix must be square');
end
p = inputParser;
defaultFontSize = 10;
defaultAngle = 30;
defaultLabels = cell(N,1);
for n = 1:N
    defaultLabels{n} = num2str(n);
end
defaultColormap = 'gray';
defaultToggle = false;
defaultNormalize = true;
defaultLegend = true;
addOptional(p,'FontSize',defaultFontSize,@isnumeric);
addOptional(p,'Angle',defaultAngle,@isnumeric);
addOptional(p,'Labels',defaultLabels,@iscellstr);
addOptional(p,'Colormap',defaultColormap,@ischar);
addOptional(p,'Toggle',defaultToggle,@islogical);
addOptional(p,'Normalize',defaultNormalize,@islogical);
addOptional(p,'Legend',defaultLegend,@islogical);
p.parse(varargin{:});
FontSize = p.Results.FontSize;
Angle = p.Results.Angle;
Labels = p.Results.Labels;
Colormap = p.Results.Colormap;
Toggle = p.Results.Toggle;
Normalize = p.Results.Normalize;
Legend = p.Results.Legend;

if Normalize
    cm = bsxfun(@rdivide,cm,sum(cm,2));
end
colormap(Colormap);
if Toggle
    imagesc(1-cm);
else
    imagesc(cm);
end
if Legend
    for i = 1:N
        for j = 1:N
            num = 100*cm(i, j);
            if num > 50
                text(j, i,num2str(num, '%3.1f'), 'color', ones(1,3)*double(Toggle), 'HorizontalAlignment', 'center', 'fontsize',FontSize);
            elseif num > 0
                text(j, i,num2str(num, '%3.1f'), 'color', ones(1,3)*(1-double(Toggle)), 'HorizontalAlignment', 'center', 'fontsize', FontSize);
            end
        end
    end
end
% turn off the axis display
% axis off;
set(gca,'XTick',1:N);
set(gca,'YTick',1:N);
% set x-axis tick labels and rotate
set(gca, 'YTickLabel', Labels);
set(gca, 'XTickLabel', Labels(:));
% alternative: manually write y-axis tick labels
% y = get(gca, 'YTick');
% text(0.4*ones(1,N),y,Labels,'HorizontalAlignment','right','fontsize',FontSize);
rotateXLabels(gca,Angle);

end