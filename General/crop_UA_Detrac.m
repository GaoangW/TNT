function crop_UA_Detrac(gt_path, seq_name, img_folder, img_format, save_folder)

% gt_path: gtInfo.mat, X,Y,W,H
% seq_name: name of the sequence
% img_folder: input image dir
% img_format: input image format, etc, png, jpg
% save_folder: cropped image dir

load(gt_path)
margin_scale = 0.15;
resize_size = 182;
X = gtInfo.X;
Y = gtInfo.Y;
W = gtInfo.W;
H = gtInfo.H;
img_list = dir([img_folder,'\*.',img_format]);
for m = 1:length(img_list)
    img_name = img_list(m).name;
    img_path = [img_folder,'\',img_name];
    img = imread(img_path);
    img_size = size(img);
    num_id = size(H,2);
    if m>size(gtInfo.H,1)
        continue
    end
    for k = 1:num_id
        if gtInfo.H(m,k)<1
            continue
        end
        xmin = round(X(m,k));
        ymin = round(Y(m,k));
        xmax = round(X(m,k)+W(m,k)-1);
        ymax = round(Y(m,k)+H(m,k)-1);
        min_side = min(xmax-xmin,ymax-ymin);
        margin = min_side*margin_scale;
        xmin = round(max(xmin-margin,1));
        ymin = round(max(ymin-margin,1));
        xmax = round(min(xmax+margin,img_size(2)));
        ymax = round(min(ymax+margin,img_size(1)));
        crop_img = img(ymin:ymax,xmin:xmax,:);
        crop_img = imresize(crop_img, [resize_size,resize_size]);
        class_name = [seq_name,'_',fileName(k,4)];
        class_folder = [save_folder,'\',class_name];
        if exist(class_folder,'dir')<=0
            mkdir(class_folder)
        end
        id_name = [class_name,'_',fileName(m,4)];
        save_path = [class_folder,'\',id_name,'.png'];
        imwrite(crop_img, save_path);
    end
end