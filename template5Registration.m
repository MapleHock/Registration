%% Step1 get resigtration mask
% Isrc = imread(fullfile('data', 'task1', 'src', 'wave.jpg'));
Isrc = imread(fullfile('data', 'task1', 'src', 'foxWave.jpg'));
Isrc(:,:,1) = Isrc(:,:,1) + 1; % remove pure dark pixel in src image
[Ms, Ns, ~] = size(Isrc);

% center crop for src image to suit the image ratio
MNRatio = [8 5];
minPartion = floor(min(Ms / MNRatio(1), Ns / MNRatio(2)));
left = floor((Ns - minPartion * MNRatio(2)) / 2) + 1;
right = left +  minPartion * MNRatio(2) - 1;
top = floor((Ms - minPartion * MNRatio(1)) / 2) + 1;
down = top +  minPartion * MNRatio(1) - 1;
IsrcCrop = Isrc(top:down, left:right,:);
[Ms, Ns, ~] = size(IsrcCrop);
xs1 = [1, Ns, 1, Ns]';
ys1 = [1, 1, Ms, Ms]';


Itarget = imread(fullfile('data', 'task1', 'template', 'targetImage5.jpg'));
[Mt, Nt, ~] = size(Itarget);

% imshow(Itarget);
% [xs2,ys2] = ginput(4);
xs2 = [298, 488, 441, 624]';
ys2 = [529, 437, 825, 735]';

tform = fitgeotrans([xs1 ys1],[xs2 ys2],'projective');

src_registered = imwarp(IsrcCrop, tform,'OutputView',imref2d(size(Itarget)));
maskReg = sum(src_registered,3) ~= 0;
imshow(maskReg)

%% Step2 get segmentation mask
maskSeg = false(Mt, Nt);


%% Step3 put the src Img
mask = maskReg & ~maskSeg;
idx = find(mask); % add segement
newI = Itarget;
newI(idx) = src_registered(idx);
newI(idx+Mt*Nt) = src_registered(idx+Mt*Nt);
newI(idx+2*Mt*Nt) = src_registered(idx+2*Mt*Nt);
figure,imshow(newI)