%% Step1 get resigtration mask
close all
% Isrc = imread(fullfile('data', 'task1', 'src', 'sketch.jpg'));
Isrc = imread(fullfile('data', 'task1', 'src', 'foxSketch.jpg'));
Isrc(:,:,1) = Isrc(:,:,1) + 1; % remove pure dark pixel in src image
[Ms, Ns, ~] = size(Isrc);

% center crop for src image to suit the image ratio
MNRatio = [7 6];
minPartion = floor(min(Ms / MNRatio(1), Ns / MNRatio(2)));
left = floor((Ns - minPartion * MNRatio(2)) / 2) + 1;
right = left +  minPartion * MNRatio(2) - 1;
top = floor((Ms - minPartion * MNRatio(1)) / 2) + 1;
down = top +  minPartion * MNRatio(1) - 1;
IsrcCrop = Isrc(top:down, left:right,:);
[Ms, Ns, ~] = size(IsrcCrop);
xs1 = [1, Ns, 1, Ns]';
ys1 = [1, 1, Ms, Ms]';


Itarget = imread(fullfile('data', 'task1', 'template', 'targetImage4.jpg'));
[Mt, Nt, ~] = size(Itarget);

% imshow(Itarget);
% [xs2,ys2] = ginput(4);
xs2 = [650, 940, 629, 910]';
ys2 = [285, 295, 643, 658]';

tform = fitgeotrans([xs1 ys1],[xs2 ys2],'projective');

src_registered = imwarp(IsrcCrop, tform,'OutputView',imref2d(size(Itarget)));
maskReg = sum(src_registered,3) ~= 0;
imshow(maskReg)

%% Step2 get segmentation mask
Ifeature = rgb2lab(Itarget);

[L, num, centerFeatures] = mySLIC(Ifeature, 500, .1);
tic 
disp('doing SLIC post process..')
L = postProcess(L);
toc
% pretest lazy snapping parameter
foreInd = [4.7435000e+05;
           5.5187500e+05;
           5.8285800e+05;
           7.0699400e+05;
           7.5129700e+05];
backInd = [ 5.3847000e+05;
           5.6931500e+05;
           6.8942100e+05;
           7.0277900e+05;
           7.5534200e+05
            ];
foreMask = false(size(Itarget, 1), size(Itarget, 2));
foreMask(foreInd) = true;
backMask = false(size(Itarget, 1), size(Itarget, 2));
backMask(backInd) = true;

maskSeg = lazysnapping(Itarget, L, foreMask, backMask);
maskSeg = imopen(maskSeg, strel('disk',30,8));
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