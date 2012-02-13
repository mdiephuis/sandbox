% Small unit test with ACF peak based regid registration
%
%
%
function so;

close all;

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


