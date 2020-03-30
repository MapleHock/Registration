%I = imread(fullfile('data', 'task1', 'src','sourceImage.jpg'));
I = imread(fullfile('data', 'task1', 'src','fox.jpg'));
I = im2double(I);
[M,N,~] = size(I);
%% Step1 get the anaglyph marginal Image

I = imfilter(I, fspecial('gaussian', 10, 10));
% get gradient in three channel, correctly evaulate the margin in color
% image
[Rpx, Rpy] = gradient(I(:,:,1));
[Gpx, Gpy] = gradient(I(:,:,2));
[Bpx, Bpy] = gradient(I(:,:,3));
   
RG = sqrt(Rpx.^2 + Rpy.^2);  
GG = sqrt(Gpx.^2 + Gpy.^2);  
BG = sqrt(Bpx.^2 + Bpy.^2);  
   
RGBGradientAmp = mat2gray(RG + GG + BG);  
   

%% Step2 hist match
% reverse the marginal image, to make marginal black
reverseI = 1 - RGBGradientAmp;
% read a real sketch as template
template = imread(fullfile('data', 'task1', 'src','hisTtemplate.jfif'));
template = im2double(rgb2gray(template));
% mean var shift for the template
meanT = mean(template(:));
varT = var(template(:));
template = (template - meanT) / sqrt(varT) * 0.2 + 0.6;
sketchI = imhistmatch(reverseI, template);

%%  step3 crop and combine layers
marginWidth = 7;
sketchI = sketchI(1 + marginWidth : M - marginWidth, 1 + marginWidth: N - marginWidth);
figure;
imshow(sketchI);
colorI = cat(3, sketchI, sketchI, sketchI);
