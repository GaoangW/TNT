function [max_v,idx] = max_mat(A)

if size(A,1)>1 && size(A,2)>1
    [v1,idx1] = max(A);
    [v2,idx2] = max(v1);
    idx = [idx1(idx2),idx2];
    max_v = v2;
elseif size(A,1)==1
    [max_v,idx2] = max(A);
    idx = [1,idx2];
elseif size(A,2)==1
    [max_v,idx1] = max(A);
    idx = [idx1,1];
end
