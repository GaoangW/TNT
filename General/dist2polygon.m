function [dist,inPoly] = dist2polygon(pts,ptsPoly)

% Input: pts(n by 2), ptsPoly(m by 2)
% Output: dist(n by 1), inPoly(n by 1) indicate whether the points are
% inside the polygon.
inPoly = inpolygon(pts(:,1), pts(:,2), ptsPoly(:,1), ptsPoly(:,2));
nLine = length(ptsPoly(:,1));
distMat = zeros(size(pts,1),nLine);
for n = 1:nLine
    if n~=nLine
        distMat(:,n) = dist2line(pts, ptsPoly(n,:), ptsPoly(n+1,:));
    else
        distMat(:,n) = dist2line(pts, ptsPoly(n,:), ptsPoly(1,:));
    end
end
dist = min(distMat,[],2);