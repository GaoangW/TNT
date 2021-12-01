function MOT_to_UA_Detrac(gt_file, seq_name, save_folder, img_size)
% gt_file: MOT gt txt
% seq_name: save name
% save_folder: save dir
fileID = fopen(gt_file,'r');
A = textscan(fileID,'%d %d %d %d %d %d %d %d %d %d','Delimiter',',');
fclose(fileID);

M = [A{1},A{2},A{3},A{4},A{5},A{6},A{7},A{8},A{9}];
max_fr = max(M(:,1));
uniq_ids = unique(M(:,2));
for n = 1:length(uniq_ids)
    A{2}(M(:,2)==uniq_ids(n)) = n;
end
M(:,2) = A{2};
max_id = length(uniq_ids);
X = zeros(max_fr,max_id);
Y = zeros(max_fr,max_id);
W = zeros(max_fr,max_id);
H = zeros(max_fr,max_id);
X(M(:,1)+(M(:,2)-1)*max_fr) = M(:,3);
Y(M(:,1)+(M(:,2)-1)*max_fr) = M(:,4);
W(M(:,1)+(M(:,2)-1)*max_fr) = M(:,5);
H(M(:,1)+(M(:,2)-1)*max_fr) = M(:,6);
gtInfo.X = X;
gtInfo.Y = Y;
gtInfo.W = W;
gtInfo.H = H;

% visibility
V = zeros(size(W));
for n = 1:size(W,1)
    idx = find(W(n,:)~=0);
    if length(idx)<=1
        V(n,idx) = 1;
        continue
    end
    bbox = zeros(length(idx),4);
    bbox(:,1) = X(n,idx)';
    bbox(:,2) = Y(n,idx)';
    bbox(:,3) = W(n,idx)';
    bbox(:,4) = H(n,idx)';
    [overlapRatio,~] = overlap(bbox,bbox);
    for k = 1:length(idx)
        overlapRatio(k,k) = 0;
    end
    max_overlap = max(overlapRatio,[],1);
    V(n,idx) = 1-max_overlap;
end
gtInfo.V = V;
gtInfo.img_size = img_size;
save_path = [save_folder,'\',seq_name,'.mat'];
save(save_path,'gtInfo');
