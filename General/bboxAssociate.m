function [det_idx, gt_idx, final_overlap_mat] = bboxAssociate(det_bbox, gt_bbox, ...
    overlap_thresh, lb_thresh, mask)

% bbox = [x,y,width,height]
if isempty(gt_bbox) || isempty(det_bbox)
    det_idx = [];
    gt_idx = [];
    final_overlap_mat = [];
    return
end
N1 = size(det_bbox, 1);
N2 = size(gt_bbox, 1);
overlap_mat = overlap(det_bbox, gt_bbox);
if ~isempty(mask)
    overlap_mat(mask==0) = 0;
end
final_overlap_mat = overlap_mat;

% greedy search
gt_idx = [];
det_idx = [];
while 1
    [max_v, idx] = max_mat(overlap_mat);
    if max_v<overlap_thresh
        break
    end
    
    if isempty(lb_thresh)
        det_idx = [det_idx,idx(1)];
        gt_idx = [gt_idx,idx(2)];
    else
        overlap_idx = find(overlap_mat(idx(1),:)>lb_thresh);
        if length(overlap_idx)==1
            det_idx = [det_idx,idx(1)];
            gt_idx = [gt_idx,idx(2)];
        end
    end
    overlap_mat(idx(1),:) = 0;
    overlap_mat(:,idx(2)) = 0;
end