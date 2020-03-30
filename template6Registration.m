%% Step1 get resigtration mask
% Isrc = imread(fullfile('data', 'task1', 'src', 'wave.jpg'));
Isrc = imread(fullfile('data', 'task1', 'src', 'foxWave.jpg'));
Isrc(:,:,1) = Isrc(:,:,1) + 1; % remove pure dark pixel in src image
[Ms, Ns, ~] = size(Isrc);

% center crop for src image to suit the image ratio
MNRatio = [1 1];
minPartion = floor(min(Ms / MNRatio(1), Ns / MNRatio(2)));
left = floor((Ns - minPartion * MNRatio(2)) / 2) + 1;
right = left +  minPartion * MNRatio(2) - 1;
top = floor((Ms - minPartion * MNRatio(1)) / 2) + 1;
down = top +  minPartion * MNRatio(1) - 1;
IsrcCrop = Isrc(top:down, left:right,:);
[Ms, Ns, ~] = size(IsrcCrop);
xs1 = [1, Ns, 1, Ns]';
ys1 = [1, 1, Ms, Ms]';


Itarget = imread(fullfile('data', 'task1', 'template', 'targetImage6.jpg'));
[Mt, Nt, ~] = size(Itarget);

% imshow(Itarget);
% [xs2,ys2] = ginput(4);
xs2 = [318, 543, 828, 1039]';
ys2 = [304, -30, 676, 273]';

tform = fitgeotrans([xs1 ys1],[xs2 ys2],'projective');

src_registered = imwarp(IsrcCrop, tform,'OutputView',imref2d(size(Itarget)));
maskReg = sum(src_registered,3) ~= 0;
imshow(maskReg)

%% Step2 get segmentation mask
Ifeature = rgb2lab(Itarget);

[L, num, centerFeatures] = mySLIC(Ifeature, 500, 1);
tic 
disp('doing SLIC post process..')
L = postProcess(L);
toc
% pretest lazy snapping parameter
foreInd = [ 4.2230600e+05;
           4.7537100e+05;
           5.0436500e+05;
           5.2578200e+05;
           5.3483800e+05;
           6.3849700e+05;
           6.9313100e+05;
           6.9507000e+05;];
backInd = [ 2.3770200e+05;
           2.5246900e+05;
           2.9114000e+05;
           3.2419500e+05;
           3.6293800e+05;
           3.7022500e+05;
           3.8468100e+05;
           3.9157000e+05;
           3.9209700e+05;
           8.4390200e+05;
           8.8948600e+05;
           9.7311600e+05;];
foreMask = false(size(Itarget, 1), size(Itarget, 2));
foreMask(foreInd) = true;
backMask = false(size(Itarget, 1), size(Itarget, 2));
backMask(backInd) = true;

maskSeg = lazysnapping(Itarget, L, foreMask, backMask);
maskSeg = imclose(maskSeg, strel('disk',20,8));
maskSeg = imopen(maskSeg, strel('disk',20,8));
maskSeg = imerode(maskSeg, strel('disk', 4,8));
maskSeg = imclose(maskSeg, strel('disk',20,8));
imshow(maskSeg)

%% Step3 fourier descriptor find ellipse
bd = bwboundaries(maskSeg);
z = frdescp(bd{1});
newBd = ifrdescp(z, 8);
boundaryMask = false(size(Itarget,1), size(Itarget, 2));
for i = 1 : length(newBd(:,1))
    boundaryMask(round(newBd(i,1)), round(newBd(i,2))) = true;
end

rebuildMask = imdilate(boundaryMask, strel('square',10));
rebuildMask = imerode(rebuildMask, strel('square',8));
imLabel = bwlabel(~rebuildMask);
stats = regionprops(imLabel,'Area');
[b,index] = sort([stats.Area],'descend');
rebuildMask = ismember(imLabel,index(2));
maskSeg = rebuildMask;
imshow(maskSeg),title('rebulid mask by sparse fourier descriptor')

%% Step4 put the src Img
mask = maskReg & maskSeg;
imshow(mask)
idx = find(mask); % add segement
newI = Itarget;
newI(idx) = src_registered(idx);
newI(idx+Mt*Nt) = src_registered(idx+Mt*Nt);
newI(idx+2*Mt*Nt) = src_registered(idx+2*Mt*Nt);
figure,imshow(newI)


%% support function from textbook
function z = frdescp(s)
%FRDESCP Computes Fourier descriptors.
%   Z = FRDESCP(S) computes the Fourier descriptors of S, which is an
%   np-by-2 sequence of image coordinates describing a boundary.  
%
%   Due to symmetry considerations when working with inverse Fourier
%   descriptors based on fewer than np terms, the number of points in
%   S when computing the descriptors must be even.  If the number of
%   points is odd, FRDESCP duplicates the end point and adds it at
%   the end of the sequence. If a different treatment is desired, the
%   sequence must be processed externally so that it has an even
%   number of points.
%
%   See function IFRDESCP for computing the inverse descriptors. 

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.4 $  $Date: 2003/10/26 23:13:28 $

% Preliminaries
[np, nc] = size(s);
if nc ~= 2 
   error('S must be of size np-by-2.'); 
end
if np/2 ~= round(np/2);
   s(end + 1, :) = s(end, :);
   np = np + 1;
end

% Create an alternating sequence of 1s and -1s for use in centering
% the transform.
x = 0:(np - 1);
m = ((-1) .^ x)';
 
% Multiply the input sequence by alternating 1s and -1s to
% center the transform.
s(:, 1) = m .* s(:, 1);
s(:, 2) = m .* s(:, 2);
% Convert coordinates to complex numbers.
s = s(:, 1) + i*s(:, 2);
% Compute the descriptors.
z = fft(s);
end

function s = ifrdescp(z, nd)
%IFRDESCP Computes inverse Fourier descriptors.
%   S = IFRDESCP(Z, ND) computes the inverse Fourier descriptors of
%   of Z, which is a sequence of Fourier descriptor obtained, for
%   example, by using function FRDESCP.  ND is the number of
%   descriptors used to computing the inverse; ND must be an even
%   integer no greater than length(Z).  If ND is omitted, it defaults
%   to length(Z).  The output, S, is a length(Z)-by-2 matrix containing
%   the coordinates of a closed boundary.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.6 $  $Date: 2004/11/04 22:32:04 $

% Preliminaries.
np = length(z);
% Check inputs.
if nargin == 1 | nd > np 
   nd = np; 
end

% Create an alternating sequence of 1s and -1s for use in centering
% the transform.
x = 0:(np - 1);
m = ((-1) .^ x)';

% Use only nd descriptors in the inverse.  Since the 
% descriptors are centered, (np - nd)/2 terms from each end of 
% the sequence are set to 0.  
d = round((np - nd)/2); % Round in case nd is odd.
z(1:d) = 0;
z(np - d + 1:np) = 0;
% Compute the inverse and convert back to coordinates.
zz = ifft(z);
s(:, 1) = real(zz);
s(:, 2) = imag(zz);
% Multiply by alternating 1 and -1s to undo the earlier 
% centering.
s(:, 1) = m.*s(:, 1);
s(:, 2) = m.*s(:, 2);
end