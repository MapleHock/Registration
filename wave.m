% % Create a wave from center image by a input rgb image I
% I = imread(fullfile('data', 'task1', 'src','sourceImage.jpg'));
I = imread(fullfile('data', 'task1', 'src','fox.jpg'));
I = im2double(I);
[M, N, ~] = size(I);
OriginalCoor = zeros(M, N, 2);

%% Step 1 get Transfermation field
centerM = ceil(M / 2);
centerN = ceil(N / 2);
[nImg, mImg] = meshgrid(1:N,1:M);
rImg = ((nImg - centerN).^2 + (mImg - centerM).^2).^0.5;

A = 0.1012;
maxR = max(centerM, centerN);
k = 10;
phi = pi / 2 + .1;
thetaImg = A * sin(2 * pi * k * rImg ./ maxR + phi);

OrigianlCoor(:,:,1) = (mImg-centerM) .* cos(thetaImg) + (nImg-centerN) .* sin(thetaImg) + centerM;
OrigianlCoor(:,:,2) = -(mImg - centerM) .* sin(thetaImg) + (nImg - centerN) .* cos(thetaImg) + centerN;

mask = rImg < maxR * 1.2;

%% Step 2 Set pixel by interpolation

newI = zeros(M, N, 3);
for i = 1 : M
    for j = 1 : N
        if (~mask(i,j))
            newI(i,j,:) = I(i,j,:);
            continue
        end
        orii = round(OrigianlCoor(i,j,1));
        orij = round(OrigianlCoor(i,j,2));
        if (orii > 0 && orii <= M && orij > 0 && orij <= N)
            newI(i,j,:) = I(orii,orij,:);
        else
            newI(i,j,:) = I(i,j,:);
        end
    end
end

imshow(newI);