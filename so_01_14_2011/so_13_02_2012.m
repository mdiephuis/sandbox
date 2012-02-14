% Small unit test with feature based registration
%
% Maurits Diephuis, Fokko Beekhof	
%
% Uses code and functions from	
%
% P. D. Kovesi.   MATLAB and Octave Functions for Computer Vision and Image Processing.
% Centre for Exploration Targeting
% School of Earth and Environment
% The University of Western Australia.
% Available from:
% http://www.csse.uwa.edu.au/~pk/research/matlabfns/. 
%
% VLFEAT toolbox, by A. Vedaldi and B. Fulkerson
% www.vlfeat.org
%
%
function so_13_02_2012;

close all;

% Read in images
im1 = imread('ruaa_1.png');
im2 = imread('ruaa_2.png');

	
% VLFEAT sift point determination
[f1, d1] = vl_sift(single(im1), 'PeakThresh', 0, 'edgethresh', 10);
[f2, d2] = vl_sift(single(im2), 'PeakThresh', 0, 'edgethresh', 10);

% Match the descriptor vectors via exhaustive nearest neighbor search
threshold = 1.5;
[match, scores] = vl_ubcmatch(d1, d2, threshold);

% Show the intial matches between the two images
m_plotMatchLines(im1, im2, f1, f2, match);

% Build a new dataset from initial matches
M1 = [f1(1, match(1, :)); f1(2, match(1, :)); ones(1, length(match))];
M2 = [f2(1, match(2, :)); f2(2, match(2, :)); ones(1, length(match))];

% Apply RANSAC to find the affine transformation
t = 0.01;											%	Distance threshold for deciding outliers  
[H, inliers] = ransacfithomography(M2, M1, t);		%	RANSAC

% RANSAC results
inPercentage = round(100*length(inliers)/length(M1));
fprintf('Number of inliers is %d (%d%%) \n', length(inliers), inPercentage);

% Show RANSAC results
m_plotMatchLines(im1, im2, [M1(1:2, :)], [M2(1:2, :)], repmat(inliers, [2 1]));

% Determine scale and rotation
[scale theta] = f_estimate_scaling_rotation(M2(1:2, inliers)', M1(1:2, inliers)');

fprintf('Found scale: %2.4f\n', scale);
fprintf('Found angle: %2.4f\n', theta);

% Optimization
scale_cos_theta = scale*cos(theta);
scale_sin_theta = scale*sin(theta);

% Build affine matrix
M_correct = [scale_cos_theta, -scale_sin_theta; scale_sin_theta,  scale_cos_theta];		
A2 = [scale*cos(theta) scale*sin(theta) 0; -scale*sin(theta) scale*cos(theta) 0; 0 0 1];
TFORM = maketform('affine', A2);

% Correct scaling and rotation		
transIm	= imtransform(im2, TFORM, 'nearest', 'XData', [1 size(im1, 2)], 'YData', [1 size(im1, 1)]);	

% Correct translation, this not done based on features
[y_offset, x_offset] = m_translation_offset(im1, transIm);
fprintf('Found y-x offset is %d by %d\n', y_offset, x_offset);
	
% Final translation correction
transIm = circshift(transIm, [-y_offset -x_offset]);

% Show and tell
figure;imagesc(im2-im1);title('original');
figure;imagesc(im1-transIm);title('registered');





% -----------------------------------------------------------------------------
%
% Example of a rigid registration of a translated image using the maximum peak
% of the cross correlation between the image and the distorted image
%
function m_example()


% Read in images
im_org = im2double(imread('cameraman.tif'));

% Original + padding
im1 = padarray(im_org, [20 35]);

% Translated copy by [20 35]
im2 = zeros(size(im1));
im2(1:size(im_org, 1), 1:size(im_org, 2)) = im_org;

figure;imshow(im1);title('original');
figure;imshow(im2);title('translated');

% Correct translation, this not done based on features
[y_offset, x_offset] = m_translation_offset(im1, im2);
fprintf('Found y-x offset is %d by %d\n', y_offset, x_offset);
	
% Final translation correction
transIm = circshift(im2, [-y_offset -x_offset]);

% Show and tell
figure;imagesc(im1-im2);title('original difference');
figure;imagesc(im1-transIm);title('difference after registration');colorbar;


