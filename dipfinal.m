clc;
clear all;
close all;

% Reading the image
[inp1, pathname] = uigetfile('*.jpg');
if isequal(inp1, 0)
    disp('User selected Cancel');
    return;
else
    disp(['User selected ', fullfile(pathname, inp1)]);
end

b = imread(fullfile(pathname, inp1));

% Sharpen the image
sharpenedImage = imsharpen(b, 'Radius', 2, 'Amount', 1);

% Display the original and sharpened images
figure, imshow(b);
title('Original Image');

figure, imshow(sharpenedImage);
title('Sharpened Image');

% Use the sharpened image for further processing
b = sharpenedImage;

% Extract color channels
r = b(:,:,1);
g = b(:,:,2);
bb = b(:,:,3);

% Disc segmentation
ne = g > 130;
binaryImage = ne;
binaryImage = imclearborder(binaryImage);
fill = imfill(binaryImage, 'holes');
se = strel('disk', 6);
dil = imdilate(fill, se);
figure, imshow(dil);
title('Disk Image');

% Cup segmentation
ne = g > 140;
binaryImage = ne;
binaryImage = imclearborder(binaryImage);
cup = imfill(binaryImage, 'holes');
se1 = strel('disk', 2);
di = imdilate(cup, se1);
cup = di;
figure, imshow(cup);
title('Cup Image');

% Calculate CDR
c1 = bwarea(dil);
c2 = bwarea(di);
cdr = c2 / c1;

% Plot the circles
a = dil;
stats = regionprops(double(a), 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
centers = stats.Centroid;
diameters = mean([stats.MajorAxisLength stats.MinorAxisLength], 2);
radii = diameters / 2;

figure, imshow(b);
hold on;
viscircles(centers, radii);
hold off;

figure;
subplot(3, 3, 1);
imshow(b);
title('Input Image');

subplot(3, 3, 2);
imshow(dil, []);
title('Disk Segment Image');

subplot(3, 3, 3);
imshow(b);
hold on;
viscircles(centers, radii);
hold off;
title('Disc Boundary');

subplot(3, 3, 4);
imshow(di, []);
title('Cup Image');

subplot(3, 3, 5);
imshow(b);
hold on;
viscircles(centers, radii / 2);
hold off;
title('Cup Boundary');

% Determine the diagnosis
if cdr < 0.45
    msgbox('NO GLAUCOMA DETECTED');
elseif cdr < 0.6 && cdr >= 0.45
    msgbox('GLAUCOMA DETECTED - MEDIUM RISK');
elseif cdr >= 0.6
    msgbox('GLAUCOMA DETECTED - HIGH RISK');
end

% Feature extraction for SVM classification
cd Database;
DF = [];

for ii = 1:14
    str = int2str(ii);
    str = strcat(str, '.jpg');
    bb = imread(str);
    
    % GLCM features
    glcms = graycomatrix(rgb2gray(bb));
    stats = graycoprops(glcms, 'Contrast', 'Correlation');
    stats1 = graycoprops(glcms, 'Energy', 'Homogeneity');
    conts = stats.Contrast;
    corre = stats.Correlation;
    en = stats1.Energy;
    ho = stats1.Homogeneity;
    
    feat = [conts corre en ho];
    DF = [DF; feat];
end

cd Database;

bbb = b;
glcms = graycomatrix(rgb2gray(bbb));
stats = graycoprops(glcms, 'Contrast', 'Correlation');
stats1 = graycoprops(glcms, 'Energy', 'Homogeneity');
conts = stats.Contrast;
corre = stats.Correlation;
en = stats1.Energy;
ho = stats1.Homogeneity;

QF = [conts corre en ho];

% SVM classification
train = DF;
xdata = train;
TrainingSet = double(xdata);
GroupTrain = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 3; 3];
TestSet = double(QF);
u = unique(GroupTrain);
numClasses = length(u);
result = zeros(length(TestSet(:, 1)), 1);

% Build models
for k = 1:numClasses
    G1vAll = (GroupTrain == u(k));
    models(k) = fitcsvm(TrainingSet, G1vAll);
end

% Classify test cases
for j = 1:size(TestSet, 1)
    for k = 1:numClasses
        if predict(models(k), TestSet(j, :))
            break;
        end
    end
    result(j) = k;
end

% Calculate accuracy
correctPredictions = sum(result == GroupTest);
accuracy = correctPredictions / numTestSamples;

disp(['Accuracy: ', num2str(accuracy * 100), '%']);