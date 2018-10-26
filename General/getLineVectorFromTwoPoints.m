function lineVec = getLineVectorFromTwoPoints(pt1, pt2)

% Input: end pts of a line
% Output: line vector
if pt1(1)~=pt2(1)
    k = (pt2(2)-pt1(2))/(pt2(1)-pt1(1));
    b = pt1(2)-k*pt1(1);
    lineVec = [k;-1;b];
else
    lineVec = [1;0;-pt1(1)];
end