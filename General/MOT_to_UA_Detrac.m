function MOT_to_UA_Detrac(gt_file, seq_name, save_folder)
% gt_file: MOT gt txt
% seq_name: save name
% save_folder: save dir
fileID = fopen(gt_file,'r');
A = textscan(fileID,'%d %d %d %d %d %d %d %d %f','Delimiter',',');
fclose(fileID);

M = [A{1},A{2},A{3},A{4},A{5},A{6},A{7},A{8},A{9}];
max_fr = max(M(:,1));
max_id = max(M(:,2));
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
save_path = [save_folder,'\',seq_name,'.mat'];
save(save_path,'gtInfo');