% Small unit test for image hash and descriptor functions
% for this SO question
% http://dsp.stackexchange.com/questions/1564/good-metric-for-qualitatively-comparing-image-patches
%
%
% Maurits Diephuis
%
% Uses:
% VLFEAT toolbox, by A. Vedaldi and B. Fulkerson
% www.vlfeat.org
%
%
function so_25_02_2012;

close all;

% Read in images
A1 	= im2double(rgb2gray(imread('images/A.png')));
B1 	= im2double(rgb2gray(imread('images/B1.png')));
B2 	= im2double(rgb2gray(imread('images/B2.png')));


% Euclidian distance
[inter, intra] = statTest(A1, B1, B2, @m_edist, []);
fprintf('Euclidian distance, inter: %2.2f, intra: %2.2f\n', inter, intra);

% Normalized CC
[inter, intra] = statTest(A1, B1, B2, @m_crosscor, []);
fprintf('Norm xcorr, inter: %2.2f, intra: %2.2f\n', inter, intra);

% Local versus Global mean
[inter, intra] = statTest(A1, B1, B2, @m_hamming, @m_hash1);
fprintf('Local mean hash, inter: %2.2f, intra: %2.2f\n', inter, intra);

% DCT based hash
[inter, intra] = statTest(A1, B1, B2, @m_hamming, @m_hash2);
fprintf('DCT, inter: %2.2f, intra: %2.2f\n', inter, intra);


% RP based hash
[inter, intra] = statTest(A1, B1, B2, @m_hamming, @m_hash3);
fprintf('RP, inter: %2.2f, intra: %2.2f\n', inter, intra);


% RP based hash
[inter, intra] = statTest(A1, B1, B2, @m_edist, @m_siftdesc);
fprintf('SIFT desc, inter: %2.2f, intra: %2.2f\n', inter, intra);



% ---------------------------------------------------------------------------	
% Wrapper for pseudo metric testing for intra - inter distance
%
%
function [inter, intra] = statTest(A1, B1, B2, metric, hash)

	if ( isempty(hash) )
		inter1 	= metric(A1, B1);
		inter2 	= metric(A1, B2);
		intra	= metric(B1, B2);

		inter = (inter1+inter2) / 2;
	else
		inter1 	= metric(hash(A1), hash(B1));
		inter2 	= metric(hash(A1), hash(B2));
		intra	= metric(hash(B1), hash(B2));

		inter = (inter1+inter2) / 2;
	end

% ---------------------------------------------------------------------------	
% Euclidean distance
%
function d = m_edist(a, b)
	a = a(:);
	b = b(:);
	d = sqrt(sum((a - b).^2));


% ---------------------------------------------------------------------------	
% Normalized cc
%
function cmax = m_crosscor(a, b)
	d = normxcorr2(a, b);
	cmax = max(d(:));


% ---------------------------------------------------------------------------	
% Hamming distance
%
function h = m_hamming(a, b)

	a = a(:);
	b = b(:);

	h = sum(xor(a, b));


% ---------------------------------------------------------------------------	
% Local vs global mean
% no sliding / sliding
% square ROI
%
function h = m_hash1(im)

	% hard wired settings
	dim = [2 2];
	slideFlag = 1;

	% Global mean
	mu = mean(im(:));

	
	if ( slideFlag == 0)
		data = im2col(im, [dim(1) dim(2)], 'distinct');
	else
		data = im2col(im, [dim(1) dim(2)], 'sliding');
	end

	% For each roi, look if the local mean is larger or smaller than
	% the local mean. Write out a 0, or 1 occordingly per block
	for i = 1:1:size(data, 2)
		tmp = mean(data(:, i));
		h(i) = tmp > mu;
	end

	h = h';


%-----------------------------------------------------------------------
%	Hash 2
%	1) reduce to grayscale
%	2) crush down to 16x16 value
%	3) DCT
%	4) Keep top left [8 8] DCT components
%	5) Binarize DCT based on sign of that same [8 8] block
%	6) Write out the binary string, or 63 bit integer ( nuke the DC)
%
function [h] = m_hash2(im_org)
	
	% Crush it down to 16x16 pixels
	thumb = imresize(im_org, [16 16]);

	dct_im = dct2(thumb);
	dct_im = dct_im(1:8, 1:8);
	
	% Binarize
	bw = dct_im >= 0;

	% Remove the DC component
	bw(1) = [];

	% Write out
	h = bw(:);




%-----------------------------------------------------------------------
%	Hash 3
%	1) Reduct to [16 16]
%	3) Blur
%	5) Random Projections
%	6) Binarize based on sign
%
function [h] = m_hash3(im)

	% Crush it down to 16x16 pixels
	thumb = imresize(im, [16 16]);

	% Blur a bit, can be removed if one so wishes. 
	H = fspecial('disk', 1);	
	thumb = imfilter(thumb, H, 'replicate');
    
	% Generate Random Projection Matrix
	L = 16;
	N = 2;
	rp = m_rp(L, N);

	% Projection
	rp_soft = double(thumb)*rp;

	% Binarize
	h = rp_soft > 0;
	h = h(:);


%-----------------------------------------------------------------------
%	SIFT descriptor
%
%
function [d] = m_siftdesc(im)

	% Define the SIFT descriptor frame
	% positon is (10, 10), scale = 1, rotation = 0
	% Note, bigger scales (blurring!) helps the performance

	fc = [10;10; 2; 0];
	
	% Run SIFT
	[~, d] = vl_sift(single(im), 'frames', fc);

