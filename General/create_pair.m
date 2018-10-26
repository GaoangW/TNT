function create_pair(dataset_dir, save_dir, num_pair, n_fold)

class_list = dir(dataset_dir);
class_list(1:2) = [];
classNum = length(class_list);
instanceNum = zeros(1,classNum);
for n = 1:classNum
    temp_dir = [dataset_dir,'\',class_list(n).name,'\*.png'];
    sub_list = dir(temp_dir);
    instanceNum(n) = length(sub_list);
end


fileID = fopen(save_dir,'w');


K = n_fold;         % 10
pairNum = num_pair; % 300
for k = 1:K
    for n = 1:pairNum
        d = 0;
        while d<1
            temp_num = 0;
            while temp_num<2
                rand_class = randi(classNum);
                class_name = class_list(rand_class).name;
                img_list = dir([dataset_dir,'\',class_name,'\*.png']);
                temp_num = instanceNum(rand_class);
            end
            choose_idx = randperm(instanceNum(rand_class),2);
            temp_name1 = img_list(choose_idx(1)).name;
            temp_name2 = img_list(choose_idx(2)).name;
            dot_loc1 = find(temp_name1=='.');
            dot_loc2 = find(temp_name2=='.');
            idx1 = str2double(temp_name1(dot_loc1-4:dot_loc1-1));
            idx2 = str2double(temp_name2(dot_loc2-4:dot_loc2-1));
            d = abs(idx1-idx2);
        end
        fprintf(fileID,'%s %d %d\n',class_name,idx1,idx2);
    end
    
    for n = 1:pairNum
        rand_class = randperm(classNum,2);
        class_name1 = class_list(rand_class(1)).name;
        class_name2 = class_list(rand_class(2)).name;
        choose_idx1 = randperm(instanceNum(rand_class(1)),1);
        choose_idx2 = randperm(instanceNum(rand_class(2)),1);
        img_list1 = dir([dataset_dir,'\',class_name1,'\*.png']);
        img_list2 = dir([dataset_dir,'\',class_name2,'\*.png']);
        temp_name1 = img_list1(choose_idx1).name;
        temp_name2 = img_list2(choose_idx2).name;
        dot_loc1 = find(temp_name1=='.');
        dot_loc2 = find(temp_name2=='.');
        idx1 = str2double(temp_name1(dot_loc1-4:dot_loc1-1));
        idx2 = str2double(temp_name2(dot_loc2-4:dot_loc2-1));
        fprintf(fileID,'%s %d %s %d\n',class_name1,idx1,class_name2,idx2);
    end
end

fclose(fileID);