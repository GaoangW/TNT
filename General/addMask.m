function img2 = addMask(img, mask, center)

% add mask to img
img2 = img;
[M,N] = size(img);
[m,n] = size(mask);
r1 = ceil(center(2)-m/2);
r2 = r1+m-1;
c1 = ceil(center(1)-n/2);
c2 = c1+n-1;

r_min = max(r1,1);
r_max = min(r2,M);
c_min = max(c1,1);
c_max = min(c2,N);

r3 = r_min-r1+1;
r4 = r_max-r1+1;
c3 = c_min-c1+1;
c4 = c_max-c1+1;
img2(r_min:r_max,c_min:c_max) = img(r_min:r_max,c_min:c_max)+mask(r3:r4,c3:c4);