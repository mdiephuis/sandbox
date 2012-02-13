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
function so;

close all;

% Read in images
im1 = imread('A.png');
im2 = imread('B.png');

[grad1, f1, d1] = m_getFeatures(im1);
[grad2, f2, d2] = m_getFeatures(im2);

% Match the descriptor vectors via exhaustive nearest neighbor search
threshold = 1.5;
[match, scores] = vl_ubcmatch(d1, d2, threshold);

% Show the intial matches between the two images
m_plotMatchLines(grad1, grad2, f1, f2, match);

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
m_plotMatchLines(grad1, grad2, [M1(1:2, :)], [M2(1:2, :)], repmat(inliers, [2 1]));

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


%----------------------------------------------------------------------------
% Converts input image to a binary image based on a hard threshold and then
% returns the gradient image and found SIFT points
%
function [grad, f, d] = m_getFeatures(im)

	% Convert to double and 0..1 range		
	im = im2double(im);

	% Hack, threshold depends on actual values, which here happen to be 0.039
	im = im > 0;

	% Invert
	im = 1- im;

	% Determine gradient
	[Dx1, Dy1] = gradient(im);
	grad = Dx1.^2 + Dy1.^2;

	% VLFEAT sift point determination
	[f, d] = vl_sift(single(grad), 'PeakThresh', 0, 'edgethresh', 10);
	

% ----------------------------------------------------------------------------
% Shows the tho images side by side with lines between matching feature points
%
%
function m_plotMatchLines(im1, im2, f1, f2, match)
	
	%	Make composite image
	im3 = m_appendImages(im1, im2);

	%	Show a figure with lines joining the matching features f1 and f2 as detemined by the indices
	% 	in the 2xn array match.
	%
	figure('Position', [100 100 size(im3, 2) size(im3, 1)]);
	colormap('gray');
	imshow(im3, []);
	hold on;
	cols1 = size(im1, 2);
	for i = 1: size(match, 2)
	    line([f1(1, match(1, i)) f2(1, match(2, i))+cols1], ...
	         [f1(2, match(1, i)) f2(2, match(2, i))], 'color', [0 0 1]);
		plot(f1(1, match(1, i)), f1(2, match(1, i)), 'g+'); hold on;
		plot(f2(1, match(2, i))+cols1, f2(2, match(2, i)), 'r+');

	end
	hold off;

% ----------------------------------------------------------------------------
%
%
function im = m_appendImages(image1, image2)

	% Select the image with the fewest rows and fill in enough empty rows
	% to make it the same height as the other image.
	rows1 = size(image1, 1);
	rows2 = size(image2, 1);

	if (rows1 < rows2)
	     image1(rows2, 1) = 0;
	else
	     image2(rows1, 1) = 0;
	end

	% Now append both images side-by-side.
	im = [image1 image2];


%------------------------------------------------------------------------------
% Determines scale and translation between point sets x and x', ie x = Ax
%
%
function [scale theta] = f_estimate_scaling_rotation(reference_points, observed_points)

	% Get sizes and verify input sanity
	[Npts   two] = size(reference_points);  assert(two == 2);
	[NObpts two] = size(observed_points );  assert(two == 2);

	assert(Npts == NObpts);

	if (numel(find(reference_points == observed_points)) == Npts*2)
	    scale = 1;
	    theta = 0;
	    return;
	end

	M = zeros(2*Npts, 2);
	y = zeros(2*Npts, 1);
	for i=1:Npts

		% Two equations per pair of points
		M(2*i-1, 1) = reference_points(i, 2);
		M(2*i-1, 2) = reference_points(i, 1);
		y(2*i-1)    =  observed_points(i, 2);
		
		M(2*i,   1) =  reference_points(i, 1);
		M(2*i,   2) = -reference_points(i, 2);
		y(2*i)      =   observed_points(i, 1);

	end


	% params(1) = scale * cos(theta); params(2) = scale * sin(theta); 
	params = inv(M'*M)*M'*y;

	% Verify 
	[two one] = size(params);
	assert(two == 2); assert(one == 1);

	% Extract scaling using (cos^2 t + sin^2 t == 1)
	scale = 1/sqrt(params(1)^2 + params(2)^2);

	% See wikipedia
	tan_theta = params(2) / params(1);
	theta = atan(tan_theta);

