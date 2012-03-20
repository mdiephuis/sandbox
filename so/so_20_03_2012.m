%
%
function so;

close all;

% Read in an image
im = imread('images/onxij_small.jpg');
im = im2double(im);

bw = f1(rgb2gray(im));
figure;imshow(bw, []);title('Method 1');

%f2(im);

%f3(im);

return;

% Method 1: Quick & dirty morphology
%
function bw = f1(im)

	% Convert to bw and deploy median filter to create a flexible 
	% threshold.
	f_makebw = @(I) im2bw(I.data, double(median(I.data(:)))/1.3);
	bw = ~blockproc(im, [128 128], f_makebw);

	% Remove cruft
	bw = bwareaopen(bw, 30);

	% Clear the border
	bw = imclearborder(bw);


% Method 2, play with HSV space
%
%
function f2(im)

	hsv_im = rgb2hsv(im);

	h_im = hsv_im(:, :, 1); 
	s_im = hsv_im(:, :, 2); 
	v_im = hsv_im(:, :, 3); 


	figure; imshow(h_im, []); title('H');
	figure; imshow(s_im, []); title('S');
	figure; imshow(v_im, []); title('V');


% Method 3, run-of-the-mill edge detectors
%
function f3(im)

	im = rgb2gray(im);

	edge1 = edge(im, 'canny', [], 1.5);
	edge2 = edge(im, 'sobel');
	edge3 = edge(im, 'log');

	figure;imshow(edge1, []); title('canny');
	figure;imshow(edge2, []); title('sobel');
	figure;imshow(edge3, []); title('log');







