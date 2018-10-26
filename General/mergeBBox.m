function [update_bbox,idx] = mergeBBox(bbox, overlap_thresh, det_score)

% bbox = [x,y,width,height].
cand_idx = ones(size(bbox,1),1);
for n1 = 1:size(bbox,1)-1
    for n2 = (n1+1):size(bbox,1)
        if cand_idx(n1)==0 || cand_idx(n2)==0
            continue
        end
        [r,overlap_area] = overlap(bbox(n1,:),bbox(n2,:));
        
%         r = overlap_area/(bbox(n1,3)*bbox(n1,4)+bbox(n2,3)*bbox(n2,4)-overlap_area);

        r1 = overlap_area/(bbox(n1,3)*bbox(n1,4));
        r2 = overlap_area/(bbox(n2,3)*bbox(n2,4));
        if isempty(det_score)
            s1 = r2;
            s2 = r1;
        else
            s1 = det_score(n1);
            s2 = det_score(n2);
        end
        
        if r1>overlap_thresh || r2>overlap_thresh
            if s1>s2
                cand_idx(n2) = 0;
            else
                cand_idx(n1) = 0;
            end
        end
        
%         if r1>=r2 && r1>overlap_thresh
%             cand_idx(n1) = 0;
%         end
%         if r1<r2 && r2>overlap_thresh
%             cand_idx(n2) = 0;
%         end
    end
end
idx = find(cand_idx==1);
update_bbox = bbox(cand_idx==1,:);