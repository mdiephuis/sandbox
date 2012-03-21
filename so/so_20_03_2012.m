%
%
function so;

close all;

% Read in an image
im = imread('images/onxij_small.jpg');
im = im2double(im);
im = rgb2gray(im);

%bw = f1(rgb2gray(im));
%figure;imshow(bw, []);title('Method 1');

%f2(im);

grid_edge = f3(im);

f5(im, grid_edge);

%f4(im);

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


% Method 3, run-of-the-mill edge detectors + houglines
%
function grid_edge = f3(im)

	edge1 = edge(im, 'canny', [], 1.5);
	
	f_makebw = @(I) im2bw(I.data, double(median(I.data(:)))/1.3);
	foreground = ~blockproc(im, [128 128], f_makebw);

	% dilate the foreground a bit
	se = strel('disk', 5);
	foreground = imdilate(foreground, se);

	edge1(find(foreground)) = 0;

	grid_edge = edge1;

	figure;imshow(edge1, []);title('edge')





% Method 4
% detect the grid
%
function f4(im)

	% Image seems to have some non-uniform lighting, so estimate the background
	background = imopen(im, strel('disk', 15));

	f_makebw = @(I) im2bw(I.data, double(median(I.data(:)))/1.3);
	foreground = ~blockproc(im, [128 128], f_makebw);

	background(find(foreground)) = 1;
	figure;imshow(background, []); title('background');
	xProfile = mean(background);

	figure;
	plot(xProfile, 'r--'); 
	grid on;

	% Unbiased autocorrelation
	ac = xcov(xProfile);

	% Left slopes
	s1 = diff(ac([1 1:end]));                   
	
	% Right slopes
	s2 = diff(ac([1:end end]));                 
	
	% Peaks
	maxima 		= find(s1>0 & s2<0);                 
	estPeriod 	= round(median(diff(maxima)))     % nominal spacing
	
	figure;
	plot(ac, 'r--'); hold on;
	plot(maxima, ac(maxima), 'r^')
	title('autocorrelation of profile')
	axis tight
	grid on;

	% Remove background morphologically, using the estimated period
	seLine = strel('line', estPeriod, 0);
	xProfile2 = imtophat(xProfile, seLine);
	
	figure;
	plot(xProfile2)
	title('enhanced horizontal profile')
	axis tight; grid on;

	level = graythresh(xProfile2/255)*255
	bw = im2bw(xProfile2/255,level/255); 
	L = bwlabel(bw);  
	
	% Locate the centers
	stats = regionprops(L);
	centroids = [stats.Centroid];
	xCenters = centroids(1:2:end);
	
	figure;
	plot(L)
	axis tight
	title('labelled regions and centers')

	hold on
	plot(xCenters, 1:max(L), 'ro')
	hold off
	

	%% Determine divisions between spots
	% The midpoints between adjacent peaks provides grid point locations.
	gap = diff(xCenters)/2;
	first = xCenters(1)-gap(1);
	xGrid = round([first xCenters(1:end)+gap([1:end end])]);
	figure
	for i=1:length(xGrid)
  		line(xGrid(i)*[1 1], ylim, 'color', 'm')
	end
	title('vertical separators')


% Method 5
% Run of the mill hough detection
%
function f5(im, im_edge)

	%	find major lines by using the Hough transform
	[H theta rho] = hough(im_edge);

	% show fancy colormap
	figure;
	hold on;
	imshow(imadjust(mat2gray(H)),'XData',theta,'YData',rho,...
	      'InitialMagnification','fit');
	title('Hough transform');
	xlabel('\theta'), ylabel('\rho');
	axis on, axis normal, hold on;
	colormap(hot);

	%	detect lines 
	hpeaks = houghpeaks(H, 45, 'Threshold', 0.025*max(H(:)), 'NHoodsize', [27 27]); 
	x = theta(hpeaks(:, 2)); 
	y = rho(hpeaks(:, 1));
	plot(x, y, 's', 'color', 'white');
	hold off;
	% extract line segments
	hlines = houghlines(im_edge, theta, rho, hpeaks, 'FillGap', 35,'MinLength', 5);

	figure;
	hold on
	imshow(im, []);
	for i=1:length(hlines)
		xy = [hlines(i).point1; hlines(i).point2];
		line(xy(:, 1),xy(:, 2), 'Color', 'g', 'LineWidth', 1);
	end;

	return


