% Demo to extract frames and get frame means from a movie and save individual frames to separate image files.
% Then rebuilds a new movie by recalling the saved images from disk.
% Also computes the mean gray value of the color channels
% And detects the difference between a frame and the previous frame.
% Illustrates the use of the VideoReader and VideoWriter classes.
% A Mathworks demo (different than mine) is located here http://www.mathworks.com/help/matlab/examples/convert-between-image-sequences-and-video.html

clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures.
clear;  % Erase all existing variables.
workspace;  % Make sure the workspace panel is showing.
fontSize = 22;

movieFullFileName = fullfile('uber_trimmed_2.mp4');
% Determine how many frames there are.
numberOfFrames = 102;
vidHeight = 720;
vidWidth = 1280;

% Create a VideoWriter object to write the video out to a new, different file.
writerObj = VideoWriter('combined_1', 'MPEG-4');
writerObj.FrameRate = 24;
open(writerObj);

% Read the frames back in from disk, and convert them to a movie.
% Preallocate recalledMovie, which will be an array of structures.
% First get a cell array with all the frames.
allTheFrames = cell(numberOfFrames,1);
allTheFrames(:) = {zeros(vidHeight, vidWidth, 3, 'uint8')};
% Next get a cell array with all the colormaps.
allTheColorMaps = cell(numberOfFrames,1);
allTheColorMaps(:) = {zeros(256, 3)};
% Now combine these to make the array of structures.
recalledMovie = struct('cdata', allTheFrames, 'colormap', allTheColorMaps)
for frame = 1 : numberOfFrames
    % Construct an output image file name.
    outputBaseFileName = sprintf('Frame %4.4d.png', frame);
    outputFullFileName = fullfile('/Users/puneet/Workspace/MatlabProjects/OpenCE-master/ying1', outputBaseFileName);
    % Read the image in from disk.
    thisFrame = imread(outputFullFileName);
    % Convert the image into a "movie frame" structure.
    recalledMovie(frame) = im2frame(thisFrame);
    % Write this frame out to a new video file.
    writeVideo(writerObj, thisFrame);
end
close(writerObj);

% Create a VideoWriter object to write the video out to a new, different file.
writerObj = VideoWriter('combined_2', 'MPEG-4');
open(writerObj);

% Read the frames back in from disk, and convert them to a movie.
% Preallocate recalledMovie, which will be an array of structures.
% First get a cell array with all the frames.
allTheFrames = cell(numberOfFrames,1);
allTheFrames(:) = {zeros(vidHeight, vidWidth, 3, 'uint8')};
% Next get a cell array with all the colormaps.
allTheColorMaps = cell(numberOfFrames,1);
allTheColorMaps(:) = {zeros(256, 3)};
% Now combine these to make the array of structures.
recalledMovie = struct('cdata', allTheFrames, 'colormap', allTheColorMaps)
for frame = 1 : numberOfFrames
    % Construct an output image file name.
    outputBaseFileName = sprintf('Frame %4.4d.png', frame);
    outputFullFileName = fullfile('/Users/puneet/Workspace/MatlabProjects/OpenCE-master/ying2', outputBaseFileName);
    % Read the image in from disk.
    thisFrame = imread(outputFullFileName);
    % Convert the image into a "movie frame" structure.
    recalledMovie(frame) = im2frame(thisFrame);
    % Write this frame out to a new video file.
    writeVideo(writerObj, thisFrame);
end
close(writerObj);
