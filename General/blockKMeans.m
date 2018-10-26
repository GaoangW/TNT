function B = blockKMeans(block_path, init_K, max_iter, disp_flag, savePath)

% Input: block_path are a cell of data path. Each cell is d by n. init_K is
% initial points or the number of points.

memory_folder = 'C:\Users\ipl333\Documents\Code\LLC dataset\memory\';

num_block = length(block_path);
num_ele = zeros(1,num_block);
blockPts = load(block_path{1});
field_name = fieldnames(blockPts);
blockPts = blockPts.(field_name{1});

dim = size(blockPts,1);
for n = 1:num_block
    blockPts = load(block_path{n});
    field_name = fieldnames(blockPts);
    blockPts = blockPts.(field_name{1});
    num_ele(n) = size(blockPts,2);
    clear blockPts
end
total_num = sum(num_ele);

% initial points
if numel(init_K)==1
    K = init_K;
    B = zeros(dim,K);
    for n = 1:K
        rand_block = randi(num_block,1);
        blockPts = load(block_path{rand_block});
        field_name = fieldnames(blockPts);
        blockPts = blockPts.(field_name{1});
        rand_ele = randi(num_ele(rand_block),1);
        B(:,n) = blockPts(:,rand_ele);
        clear blockPts
    end
else
    B = init_K;
    K = size(B,2);
end

% iteration
class_idx = zeros(1,total_num);
cluster_pts = zeros(1,K);
for n = 1:max_iter
    if disp_flag
        n
    end
    prev_B = B;
    cnt = 0;
    for k = 1:num_block
        blockPts = load(block_path{k});
        field_name = fieldnames(blockPts);
        blockPts = blockPts.(field_name{1});
        for kk = 1:num_ele(k)
            cnt = cnt+1;
            [~,max_idx] = max(B'*blockPts(:,kk));
            class_idx(cnt) = max_idx;
            cluster_pts(max_idx) = cluster_pts(max_idx)+1;
        end
        clear blockPts
    end
    
    % put pts into clusters
    code_idx = zeros(1,K);
    X = cell(1,K);
    X_paths = cell(1,K);
    for k = 1:K
%         pack
%         X{k} = zeros(dim,cluster_pts(k));   
        X{k} = sparse(dim,cluster_pts(k));   
    end
    cnt = 0;
    for k = 1:num_block
        blockPts = load(block_path{k});
        field_name = fieldnames(blockPts);
        blockPts = blockPts.(field_name{1});
        for kk = 1:num_ele(k)
            cnt = cnt+1;
            max_idx = class_idx(cnt);
            code_idx(max_idx) = code_idx(max_idx)+1;
            X{max_idx}(:,code_idx(max_idx)) = sparse(blockPts(:,kk));
        end
        clear blockPts
    end 
    
    for k = 1:K
        X_paths{k} = [memory_folder,'X',num2str(k),'.mat'];
        XX = X{k};
        save(X_paths{k},'XX');
        clear XX
    end
    clear X
    
    % update centers
    for k = 1:K
        load(X_paths{k});
        XX = full(XX);
        
        if ~isempty(XX)
            B(:,k) = mean(XX,2);
            if norm(B(:,k))~=0
                B(:,k) = B(:,k)/norm(B(:,k));
            end
        else
            rand_block = randperm(num_block,1);
            blockPts = load(block_path{rand_block});
            field_name = fieldnames(blockPts);
            blockPts = blockPts.(field_name{1});
            rand_ele = randperm(num_ele(rand_block),1);
            B(:,k) = blockPts(:,rand_ele);
            clear blockPts
        end
        
        clear XX
    end
    
%     clear X
    
    if disp_flag
        cor_v = min(sum(B.*prev_B))
    else
        cor_v = min(sum(B.*prev_B));
    end
    
    if cor_v>0.9999
        break
    end
    
    if nargin==5
        save(savePath,'B');
    end
end

