close all
V = VideoReader(fullfile('data', 'task2', 'targetVideo.MP4'));
newCover = imread(fullfile('data', 'task2', 'newCover.jpg'));

[M,N,~] = size(newCover);
coverPos = [1,1;
             N,1;
             N,M;
             1,M];

anchorFrame = readFrame(V);
[Mt, Nt, ~] = size(anchorFrame);
% f = imshow(anchorFrame);
% [xs,ys] = ginput(4);
% close gcf

xs = [64, 672, 783, 84]';
ys = [150, 106, 804, 891]';
projCoor = [xs'; ys'; ones(1,4)];

prevFrame = anchorFrame;
anchorCoor = projCoor;
i = 1;

videoName = 'replace.avi';
fps = 25;   
vWriter = VideoWriter(videoName);
vWriter.FrameRate=fps;
open(vWriter);

% the strel for close operate for fill the hand mask
seForHandMask = strel('disk', 7);
tic
while 1    
    nextFrame = readFrame(V);
    i = i + 1;
   
    % in the rotate interval, we need correct the project coordinate groups 
    % every 20 frame, total 4 times correction
    if (((i >= 230 && i < 300) && (mod(i, 20) == 10)))
       projCoor =  HoughCorr(projCoor, prevFeature);
    end
    
    % get S channel of the frames, use as SURF character match
    prevFeature = rgb2hsv(prevFrame);
    prevFeature = prevFeature(:,:,2);
    nextFeature = rgb2hsv(nextFrame);
    nextFeature = nextFeature(:,:,2);
    % get transformation matrix
    transM = getTransM(prevFeature, nextFeature);
    % update cornor point group coordnate 
    projCoor = transM * projCoor;
    prevFrame = nextFrame;
    
    
    mask = poly2mask(double(projCoor(1,:)), double(projCoor(2,:)), Mt, Nt);
    
    % correct the mask if in the interval of the hand barrying the cover
    if (i >=265 && i <= 345)
        handmask = getHandMask(nextFrame);
        handmask = handmask & mask;
        handmask = imclose(handmask, seForHandMask);      
        mask = mask & ~handmask;
    end
    
    % replace old cover by the new cover hand write to video
    tform = fitgeotrans(coverPos, projCoor(1:2,:)', 'projective');
    src_registered = imwarp(newCover, tform, 'OutputView', imref2d(size(nextFrame)));
    
    idx = find(mask);
    wFrame = nextFrame;
    wFrame(idx) = src_registered(idx);
    wFrame(idx + Mt * Nt) = src_registered(idx + Mt * Nt);
    wFrame(idx + 2 * Mt * Nt) = src_registered(idx + 2* Mt * Nt);       
    writeVideo(vWriter, wFrame);
    if ~hasFrame(V)
        break;
    end
end
toc
close(vWriter);

%% Support function 1 get transfermation matrix by SURF and MLESAC
function transM = getTransM(Isrc, Idest)
    %% Step1 get surf character
    pSrc = detectSURFFeatures(Isrc);
    pDest = detectSURFFeatures(Idest);
    [featureSrc, pSrc] = extractFeatures(Isrc, pSrc);
    [featureDest, pDest] = extractFeatures(Idest, pDest);
    
    srcCoor = pSrc.Location;
    destCoor = pDest.Location;
    
    %% Step2 sort reletively definitive point pair
    covMatrix = pdist2(featureSrc, featureDest, 'cityblock');
    [fixSrcMatrix, ind] = sort(covMatrix, 2);
    likeDontRatio = fixSrcMatrix(:,1) ./ fixSrcMatrix(:,2);
    T = 0.7;
    inliersInd = find(likeDontRatio < T);
    
    inlierSrcCoor = [srcCoor(inliersInd, 1), srcCoor(inliersInd, 2)];
    inlierDestCoor = [destCoor(ind(inliersInd), 1), destCoor(ind(inliersInd), 2)];
    trans = estimateGeometricTransform(inlierSrcCoor, inlierDestCoor,'affine');
    transM = trans.T';
end

%% Support function 2 get correct corner group by hough detection
function newCoor = HoughCorr(coor, frameFeature)
    %% Step1 get edge image by canny
    bw = edge(frameFeature,'Canny', [0 0.5],8);
    
    %% Step2 Sort edge by the corner group input
    % Sort out those edges close to the biased rectange
    mask = createRectangleMask(coor(1:2,:), size(bw), 1000);
    mask = imdilate(mask, strel('square', 100));
    bw = mask & bw;
    
    %% Step 3 Hough line detection
    [H,T,R] = hough(bw);
    P  = houghpeaks(H,8,'threshold',ceil(0.1*max(H(:))));
    lines= houghlines(bw,T,R,P,'FillGap',10,'MinLength',100);
    
    %% Step4 Sort out the line we need 
    % calcuate point - line distance
    dGroup = zeros(4, length(lines));
    for i = 1 : 4
        for j = 1 : length(lines)
            rho = lines(j).rho;
            theta = lines(j).theta;
            dGroup(i,j) = abs(coor(1,i) * cosd(theta) + coor(2, i) * sind(theta) - rho);
        end
    end
    % detele repeat line
    [dGroup, indPoint] = sort(dGroup, 1);
    minSumDist = dGroup(1,:) + dGroup(2,:);
    indPoint = indPoint(1,:) .* indPoint(2,:); %four original conrner point multi code, 2 6 12 4
    checked = false(12,1);
    [sortDist, ind] = sort(minSumDist);
    T = 60; % threshold for fake line
    coorLineInd = zeros(1,4);
    rho = zeros(4, 1);
    theta = zeros(4, 1);
    newPointCounter = 1;
    for i = 1 : length(minSumDist)
        if (sortDist(i) > T || checked(indPoint(ind(i))))
            continue
        end
        coorLineInd(newPointCounter) = ind(i);
        rho(newPointCounter) = lines(ind(i)).rho;
        theta(newPointCounter) = lines(ind(i)).theta;
        checked(indPoint(ind(i))) = true;
        newPointCounter = newPointCounter + 1;
    end
    
    %% Step5 correct corner by the intersection of those lines
    if (numel(find(coorLineInd ~= 0)) == 4)
        points = [];
        for i = 1 : 4
            for j = i + 1 : 4
                point = [cosd(theta(i)) sind(theta(i)); cosd(theta(j)) sind(theta(j))] \ [rho(i); rho(j)];
                points = [points, point];
            end
        end
        
        oldNewDist = pdist2((coor(1:2,:))', points');
        disS = sort(oldNewDist(:));
        mask = oldNewDist <= disS(4);
        [old, new] = find(mask);
        newCoor = coor;
        for i = 1 : length(old)
            if (old(i) == 4)
                continue
            end
            newCoor(1:2, old(i)) = 0.85 * points(1:2, new(i)) + 0.15 * newCoor(1:2, old(i));
        end
        
    elseif (numel(find(coorLineInd ~= 0)) == 3)
        points = [];
        for i = 1 : 3
            for j = i + 1 : 3
                point = [cosd(theta(i)) sind(theta(i)); cosd(theta(j)) sind(theta(j))] \ [rho(i); rho(j)];
                points = [points, point];
            end
        end
        
        oldNewDist = pdist2((coor(1:2,:))', points');
        disS = sort(oldNewDist(:));
        mask = oldNewDist <= disS(2);
        [old, new] = find(mask);
        newCoor = coor;
        for i = 1 : length(old)
            if (old(i) == 4 || old(i) == 1)
                continue
            end
            newCoor(1:2, old(i)) = 0.85 * points(1:2, new(i)) + 0.15 * newCoor(1:2, old(i));
        end
        % approxite correct the corner loss interaction corner by interface
        % the error vector from those corrected ones
        newCoor(1:2, 1) = newCoor(1:2, 1) + mean(newCoor(1:2,:) - coor(1:2,:), 2) .* [2;-2];
        
    end
end

%% Support function3 create rectangle mask by the corner coordinate
function mask = createRectangleMask(coor, sizeMN, numPoint)
    lambda = 0 : 1 / numPoint : 1;
    p1 = reshape(coor, [2 1 4]);
    p2 = cat(3, p1(:,:,2:4), p1(:,:,1));
    p = zeros(2, length(lambda), 4);
    ind = [];
    for i = 1 : 4
        p(:,:,i) = p1(:,:,i) * lambda + p2(:,:,i) * (1 - lambda);
        ind = [ind, sub2ind(sizeMN, round(p(2,:,i)), round(p(1,:,i)))];
    end
    mask = false(sizeMN);
    mask(ind) = true;   
end

%% Support function4 get handmask by HSV space color slic in Mahalanobis distance
function mask = getHandMask(I)
I = rgb2hsv(I);
% from teacher code, chapter 8,  colorslicing.m
% pretest hsv center a
a = [0.9570    0.2313    0.5255];
R = 60/255;
D = (I(:,:,1)-a(1)).^2+.3*(I(:,:,2)-a(2)).^2+.1*(I(:,:,3)-a(3)).^2;
mask = D<=R*R;
a = [0.0345    0.2117    0.5373];
D = (I(:,:,1)-a(1)).^2+.3*(I(:,:,2)-a(2)).^2+.1*(I(:,:,3)-a(3)).^2;
mask = mask | D<=R*R;
end
