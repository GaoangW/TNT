function [ratio_mat,overlap_area] = overlap(rect1, rect2)

% if input are rectangulars, rect = [x,y,width,height].
N1 = size(rect1,1);
N2 = size(rect2,1);
[~,xmin1] = meshgrid(1:N2,rect1(:,1));
[xmin2,~] = meshgrid(rect2(:,1),1:N1);
[~,ymin1] = meshgrid(1:N2,rect1(:,2));
[ymin2,~] = meshgrid(rect2(:,2),1:N1);
[~,xmax1] = meshgrid(1:N2,rect1(:,1)+rect1(:,3)-1);
[xmax2,~] = meshgrid(rect2(:,1)+rect2(:,3)-1,1:N1);
[~,ymax1] = meshgrid(1:N2,rect1(:,2)+rect1(:,4)-1);
[ymax2,~] = meshgrid(rect2(:,2)+rect2(:,4)-1,1:N1);
xmin = max(xmin1,xmin2);
ymin = max(ymin1,ymin2);
xmax = min(xmax1,xmax2);
ymax = min(ymax1,ymax2);
mask = (xmax>xmin & ymax>ymin);
ratio_mat = zeros(N1,N2);
overlap_area = zeros(N1,N2);
overlap_area(mask) = (xmax(mask)-xmin(mask)).*(ymax(mask)-ymin(mask));
area1 = (xmax1-xmin1+1).*(ymax1-ymin1+1);
area2 = (xmax2-xmin2+1).*(ymax2-ymin2+1);
ratio_mat(mask) = overlap_area(mask)./(area1(mask)+area2(mask)-overlap_area(mask));








% x_min = min(rect1(1),rect2(1));
% y_min = min(rect1(2),rect2(2));
% x_max = max(rect1(1)+rect1(3)-1,rect2(1)+rect2(3)-1);
% y_max = max(rect1(2)+rect1(4)-1,rect2(2)+rect2(4)-1);
% mask1 = zeros(x_max-x_min+1,y_max-y_min+1);
% mask2 = zeros(x_max-x_min+1,y_max-y_min+1);
% mask1(rect1(1)-x_min+1:rect1(1)+rect1(3)-x_min,rect1(2)-y_min+1:rect1(2)+rect1(4)-y_min) = 1;
% mask2(rect2(1)-x_min+1:rect2(1)+rect2(3)-x_min,rect2(2)-y_min+1:rect2(2)+rect2(4)-y_min) = 1;
% area1 = sum(sum(mask1));
% area2 = sum(sum(mask2));
% overlap_area = sum(sum((mask1+mask2)==2));
% ratio = 2*overlap_area/(area1+area2);