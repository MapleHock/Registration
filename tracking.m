close all
V = VideoReader(fullfile('data', 'task2', 'targetVideo.MP4'));

anchorFrame = readFrame(V);

% f = imshow(anchorFrame);
% [xs,ys] = ginput(4);
% close gcf
xs = [64, 670, 781, 84]';
ys = [155, 111, 804, 891]';
projCoor = [xs'; ys'; ones(1,4)];

prevFrame = anchorFrame;
anchorCoor = projCoor;
i = 1;

% set video writer as the same fps from src video
videoName = 'tracking.avi';
fps = 25;   
vWriter = VideoWriter(videoName);
vWriter.FrameRate=fps;
open(vWriter);

tic
while 1    
    nextFrame = readFrame(V);
    i = i + 1;
   
    % in the rotate interval, we need correct the project coordinate groups 
    % every 20 frame, total 4 times correction
    if ((i >= 230 && i < 300) && (mod(i, 20) == 10))
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
    
    % add tracking rectange and write to video
    pos = reshape(projCoor(1:2,:), 1, []);
    wFrame = insertText(insertShape(nextFrame, 'Polygon', pos, 'LineWidth', 5, 'SmoothEdges', false), projCoor(1:2,1)', 'cover', 'FontSize',18);
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
