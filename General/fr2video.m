function flag = fr2video(folder, output_video, fr_rate)

dbstop if error
img_list = dir(strcat(folder,'*.jpg'));
if isempty(img_list)
    img_list = dir(strcat(folder,'*.png'));
end
if isempty(img_list)
    img_list = dir(strcat(folder,'*.bmp'));
end
if isempty(img_list)
    flag = 1;
    disp('error: no images');
    return;
end

writerObj = VideoWriter(output_video);
writerObj.FrameRate = fr_rate;
open(writerObj);
N = length(img_list);
for n = 1:N
    fr_name = img_list(n).name;
    file_name = [folder,fr_name];
    frame = imread(file_name);
    if n==1
        img_size = size(frame);
    end
    temp_size = size(frame);
    if temp_size(1)~=img_size(1) || temp_size(2)~=img_size(2)
        frame = imresize(frame, [img_size(1),img_size(2)]);
    end
    writeVideo(writerObj,frame);
end
close(writerObj)