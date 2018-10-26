function [tr_idx, val_idx] = tr_val_split(y,split_ratio)

if size(y,2)>size(y,1)
    y = y';
end
tr_idx = [];
val_idx = [];
uniq_label = unique(y);
for n = 1:length(uniq_label)
    temp_idx = find(y==uniq_label(n));
    tr_num = ceil(length(temp_idx)*split_ratio);
    val_num = length(temp_idx)-tr_num;
    if val_num==0 && tr_num>=2
        tr_num = tr_num-1;
        val_num = val_num+1;
    end
    rand_idx = randperm(length(temp_idx),tr_num);
    idx = zeros(1,length(temp_idx));
    idx(rand_idx) = 1;
    tr_idx = [tr_idx;temp_idx(idx==1)];
    val_idx = [val_idx;temp_idx(idx==0)];
end    
tr_idx = tr_idx';
val_idx = val_idx';