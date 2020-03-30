% Create a oilpaint style image by a input rgb image I

% I = imread(fullfile('data', 'task1', 'src','sourceImage.jpg'));
I = imread(fullfile('data', 'task1', 'src', 'fox.jpg'));
[M, N, ~] = size(I);
IGray = rgb2gray(I);

% set parameter of oil paint simulation
% there are a certain number of bucket can be used in drawing
% The hist group in gray image decide which bucket a pixel belong
% and the color of a bucket will adjust adaptively
canvas = zeros(M, N, 3);
bucketNum = 10;
bucketWidth = floor(256/bucketNum);
bucketNum = ceil(256/bucketWidth);
bucketBelong = ceil(double(IGray / bucketWidth));

newI = zeros(size(I));
neighRadius = 10;
tic
for i = 1 : M
    for j = 1 : N
        % get local information,
        % use the bucket corresponding to the max frequency
        % the color of this bucket will be the mean of the pixel belongs to
        % the bucket
        
        % get local info
        rowStart = max(1, i - neighRadius);
        rowEnd = min(M, i + neighRadius);
        colStart = max(1, j - neighRadius);
        colEnd = min(N, j + neighRadius);
        localBelong = bucketBelong(rowStart:rowEnd, colStart:colEnd);
        localI = I(rowStart:rowEnd, colStart:colEnd,:);
        
        % get statistical info -> max frequency bucket num
        table = tabulate(localBelong(:));
        [freq, num] = max(table(:,2));
        num = table(num,1);
        
        % decide the color of the bucket by means
        localMask = localBelong == num;
        localInd = find(localMask);
        meanPixel = zeros(1,1,3);
        meanPixel(1) = mean(localI(localInd));
        meanPixel(2) = mean(localI(localInd + numel(localI) / 3));
        meanPixel(3) = mean(localI(localInd + numel(localI) / 3 * 2));
        newI(i,j,:) = meanPixel;
    end
end
toc
imshow(uint8(newI));
