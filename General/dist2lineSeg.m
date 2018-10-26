function d = dist2lineSeg(pts, linePt1, linePt2, lineVec, seg_flag, opt)

% Input: pts(n by 2), linePt1(1 by 2), linePt1(1 by 2)
% Output: dist to the line
if strcmp(opt,'pts')
    lineVec = getLineVectorFromTwoPoints(linePt1, linePt2);
elseif strcmp(opt,'lineVec')
    lineVec = lineVec;
end
    
data = [pts,ones(size(pts,1),1)];
d = abs(data*lineVec)/sqrt(lineVec(1)^2+lineVec(2)^2);
if seg_flag==0
    return
end
sign1 = sign([pts(:,1)-linePt1(1),pts(:,2)-linePt1(2)]*(linePt2-linePt1)')<0;
sign2 = sign([pts(:,1)-linePt2(1),pts(:,2)-linePt2(2)]*(linePt1-linePt2)')<0;
out_idx = sign1 | sign2;
if sum(out_idx)==0
    return
end
d(out_idx) = min(sqrt((pts(out_idx,1)-linePt1(1)).^2+(pts(out_idx,2)-linePt1(2)).^2),...
    sqrt((pts(out_idx,1)-linePt2(1)).^2+(pts(out_idx,2)-linePt2(2)).^2));