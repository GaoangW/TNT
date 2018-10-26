function feaSet = extrSIFT(I, gridSpacing, patchSize, nrml_threshold)

[im_h, im_w] = size(I);
feaSet.width = im_w;
feaSet.height = im_h;
feaSet.feaArr = [];
feaSet.x = [];
feaSet.y = [];

% make grid sampling SIFT descriptors
remX = mod(im_w-patchSize,gridSpacing);
offsetX = floor(remX/2)+1;
remY = mod(im_h-patchSize,gridSpacing);
offsetY = floor(remY/2)+1;
    
[gridX,gridY] = meshgrid(offsetX:gridSpacing:im_w-patchSize+1, offsetY:gridSpacing:im_h-patchSize+1);
            
% find SIFT descriptors
siftArr = sp_find_sift_grid(I, gridX, gridY, patchSize, 0.8);
[siftArr, siftlen] = sp_normalize_sift(siftArr, nrml_threshold);
            
feaSet.feaArr = [feaSet.feaArr,siftArr'];
temp_x = (gridX(:) + patchSize/2 -0.5)*(feaSet.width/im_w);
temp_y = (gridY(:) + patchSize/2 -0.5)*(feaSet.width/im_w);
feaSet.x = [feaSet.x;temp_x];
feaSet.y = [feaSet.y;temp_y];  