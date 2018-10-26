function [min_v,idx] = min_mat(A)

if size(A,1)>1 && size(A,2)>1
    [v1,idx1] = min(A);
    [v2,idx2] = min(v1);
    idx = [idx1(idx2),idx2];
    min_v = v2;
elseif size(A,1)==1
    [min_v,idx2] = min(A);
    idx = [1,idx2];
elseif size(A,2)==1
    [min_v,idx1] = min(A);
    idx = [idx1,1];
end