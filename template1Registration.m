%% Step1 get resigtration mask
close all
% Isrc = imread(fullfile('data', 'task1', 'src', 'oilpaint.jpg'));
Isrc = imread(fullfile('data', 'task1', 'src', 'foxOilpaint.jpg'));
[Ms, Ns, ~] = size(Isrc);

% center crop for src image to suit the image ratio
MNRatio = [4 3];
minPartion = floor(min(Ms / MNRatio(1), Ns / MNRatio(2)));
left = floor((Ns - minPartion * MNRatio(2)) / 2) + 1;
right = left +  minPartion * MNRatio(2) - 1;
top = floor((Ms - minPartion * MNRatio(1)) / 2) + 1;
down = top +  minPartion * MNRatio(1) - 1;
IsrcCrop = Isrc(top:down, left:right,:);
[Ms, Ns, ~] = size(IsrcCrop);
xs1 = [1, Ns, 1, Ns]';
ys1 = [1, 1, Ms, Ms]';


Itarget = imread(fullfile('data', 'task1', 'template', 'targetImage1.jpg'));
[Mt, Nt, ~] = size(Itarget);

% [xs2,ys2] = ginput(4);
xs2 = [187, 669, 185, 668]';
ys2 = [181, 183, 795, 796]';

tform = fitgeotrans([xs1 ys1],[xs2 ys2],'projective');

src_registered = imwarp(IsrcCrop, tform,'OutputView',imref2d(size(Itarget)));
maskReg = sum(src_registered,3) ~= 0;
imshow(maskReg)

%% Step2 get segmentation mask
Ifeature = rgb2lab(Itarget);

[L, num, centerFeatures] = mySLIC(Ifeature, 900, .8);
tic 
disp('doing SLIC post process..')
L = postProcess(L);
toc
% pretest lazy snapping parameter
foreInd = [ 3.8341500e+05; 3.9067500e+05; 3.9419500e+05; 3.9818100e+05;
            4.0879800e+05; 4.4114700e+05; 5.1337500e+05; 5.6006400e+05;
            5.7075600e+05; 5.7774700e+05; 5.8148700e+05; 6.3217500e+05;
            6.3566400e+05; 6.7515000e+05; 7.1831100e+05];
backInd = [ 2.9711700e+05; 3.1489600e+05; 3.1861500e+05; 4.0124500e+05;
            4.7350200e+05; 5.0217700e+05; 5.1310200e+05; 5.3451700e+05;
            6.4602100e+05; 6.5688100e+05; 6.9988900e+05; 7.1103900e+05; 
            7.2186600e+05;
            ];
foreMask = false(size(Itarget, 1), size(Itarget, 2));
foreMask(foreInd) = true;
backMask = false(size(Itarget, 1), size(Itarget, 2));
backMask(backInd) = true;

maskSeg = lazysnapping(Itarget, L, foreMask, backMask);
scarfBbottleneckPoint = [665, 360];
maskSeg(scarfBbottleneckPoint) = true;
maskSeg = imclose(maskSeg, strel('disk',16,8));
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