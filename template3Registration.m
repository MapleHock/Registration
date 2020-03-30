%% Step1 get resigtration mask
close all
% Isrc = imread(fullfile('data', 'task1', 'src', 'oilpaint.jpg'));
Isrc = imread(fullfile('data', 'task1', 'src', 'foxOilpaint.jpg'));
[Ms, Ns, ~] = size(Isrc);
xs1 = [1, Ns, 1, Ns]';
ys1 = [1, 1, Ms, Ms]';

Itarget = imread(fullfile('data', 'task1', 'template', 'targetImage3.jpg'));
[Mt, Nt, ~] = size(Itarget);

% imshow(Itarget);
% [xs2,ys2] = ginput(4);
xs2 = [349, 569, 397, 600]';
ys2 = [120, 94, 646, 540]';

tform = fitgeotrans([xs1 ys1],[xs2 ys2],'projective');

src_registered = imwarp(Isrc, tform,'OutputView',imref2d(size(Itarget)));
maskReg = sum(src_registered,3) ~= 0;
imshow(maskReg)

%% Step2 get segmentation mask
Ifeature = rgb2lab(Itarget);

[L, num, centerFeatures] = mySLIC(Ifeature, 400, .1);
tic 
disp('doing SLIC post process..')
L = postProcess(L);
toc
% a small barrier, directly choose the super pixel label of it
clipLabelInd = [4.2443200e+05];

maskSeg = (L == L(clipLabelInd(1)));
imshow(maskSeg);


%% Step3 put the src Img
mask = maskReg & ~maskSeg;
imshow(mask)
idx = find(mask); % add segement
newI = Itarget;
newI(idx) = src_registered(idx);
newI(idx+Mt*Nt) = src_registered(idx+Mt*Nt);
newI(idx+2*Mt*Nt) = src_registered(idx+2*Mt*Nt);
figure,imshow(newI)