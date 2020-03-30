%% Step1 get resigtration mask
close all
% Isrc = imread(fullfile('data', 'task1', 'src', 'sketch.jpg'));
Isrc = imread(fullfile('data', 'task1', 'src', 'foxSketch.jpg'));
Isrc(:,:,1) = Isrc(:,:,1) + 1;
[Ms, Ns, ~] = size(Isrc);

% center crop for src image
MNRatio = [9 16];
minPartion = floor(min(Ms / MNRatio(1), Ns / MNRatio(2)));
left = floor((Ns - minPartion * MNRatio(2)) / 2) + 1;
right = left +  minPartion * MNRatio(2) - 1;
top = floor((Ms - minPartion * MNRatio(1)) / 2) + 1;
down = top +  minPartion * MNRatio(1) - 1;
IsrcCrop = Isrc(top:down, left:right,:);
[Ms, Ns, ~] = size(IsrcCrop);

Itarget = imread(fullfile('data', 'task1', 'template', 'targetImage2.jpg'));
[Mt, Nt, ~] = size(Itarget);

% point pair match

xs1 = [1, Ns, 1, Ns, Ns/2, Ns / 10, Ns / 2,1,    Ns]';
ys1 = [1, 1, Ms, Ms, 1,    Ms,      Ms,    Ms/2, Ms / 2]';

xs2 = [47, 541, 20, 591, 308, 89, 300, 31, 566]';
ys2 = [50, 53, 368, 411, 49, 371, 391, 206, 222]';


Itps = putImageByTps([xs2,ys2], [xs1,ys1], [Mt,Nt], IsrcCrop);


maskReg = sum(Itps,3) ~= 0;
imshow(maskReg)
%% Step2 get segmentation mask
Ifeature = rgb2lab(Itarget);

[L, num, centerFeatures] = mySLIC(Ifeature, 700, .1);
tic 
disp('doing SLIC post process..')
L = postProcess(L);
toc
% a small barrier, directly choose the super pixel label of it
peopleLabelInd =    [2.6187700e+05; 2.6240300e+05];

maskSeg = (L == L(peopleLabelInd(1))) | (L == L(peopleLabelInd(2)));
imshow(maskSeg)

%% Step3 put the src Img
mask = maskReg & ~maskSeg;
imshow(mask)
idx = find(mask); % add segement
newI = Itarget;
newI(idx) = Itps(idx);
newI(idx+Mt*Nt) = Itps(idx+Mt*Nt);
newI(idx+2*Mt*Nt) = Itps(idx+2*Mt*Nt);
figure,imshow(newI)

function img = putImageByTps(targetPointGroup, controlPointGroup, targetSize, controlImg)
    img = zeros([targetSize, 3]);
    controlSize = size(controlImg);
    controlSize = controlSize(1:2);
    valX = controlPointGroup(:,1);
    valY = controlPointGroup(:,2);
    [a1X,axX,ayX,wX] = getTpsParameter(targetPointGroup, valX);
    [a1Y,axY,ayY,wY] = getTpsParameter(targetPointGroup, valY);
    for i = 1 : size(img,1)
        for j = 1 : size(img,2)
            oriX = tpsFunc(j, i, targetPointGroup, a1X, axX, ayX, wX);
            oriY = tpsFunc(j, i, targetPointGroup, a1Y, axY, ayY, wY);
            oriX = round(oriX);
            oriY = round(oriY);
            if (oriX < 1 || oriY < 1 || oriX > controlSize(2) || oriY > controlSize(1))
                continue
            end
            img(i, j, :) = controlImg(oriY, oriX,:);
        end
    end
    img = uint8(img);
end


function  [a1,ax,ay,w] = getTpsParameter(pts, target_val)
n = size(pts, 1);
% Create matrix
P = [ones(n, 1), pts];
K = zeros(n,n);
for i = 1 : n
    for j = 1 : n
        r = sqrt((pts(i,1) - pts(j,1)).^2 + (pts(i,2) - pts(j,2)).^2);
        K(i,j) = r^2 * log((r + eps)^2);
    end
end
A = [K P; 
    P', zeros(3)];
v = [target_val; zeros(3,1)];

% Compute and output results
coef = A \ v;
w    = coef(1:n);
a1   = coef(n + 1);
ax   = coef(n + 2);
ay   = coef(n + 3);
end

function v = tpsFunc(x, y, pts, a1, ax, ay, w)
    rGroup = sqrt((pts(:,1) - x).^2 + (pts(:,2) - y).^2);
    v = a1 + ax * x + ay * y + w' * (rGroup.^2 .* log((rGroup + eps).^2));
end
