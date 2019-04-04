function [data,count] = loadvideo(filename,downsample)
% function to read video data

if nargin < 2
    downsample = 1;
end

v = VideoReader(filename);
% Read video frames until available

count = 0;
while hasFrame(v)
    count = count + 1;    
    vidFrame = readFrame(v);
    if count == 1
        [r,c,ch] = size(vidFrame);
        data = zeros(r,c,ch,100,'uint8');
    end
    data(:,:,:,count) = vidFrame;    
end
data = data(:,:,:,1:downsample:count);
count = size(data,4);