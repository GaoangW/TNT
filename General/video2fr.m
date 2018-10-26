function flag = video2fr(video_file, output_folder, save_format, name_length)

vid = VideoReader(video_file);
numFrames = vid.NumberOfFrames;
maxLength = length(num2str(numFrames));
if maxLength>name_length
    disp('error: the length of the name is smaller than the length of the frame.')
    flag = 1;
    return
end
for n = 1:numFrames
    fr = read(vid,n);
    idx_str = num2str(n);
    idx_length = length(idx_str);
    diff_length = name_length-idx_length;
    file_name = [];
    for k = 1:diff_length
        file_name = [file_name,'0'];
    end
    file_name = [output_folder, file_name, idx_str, '.', save_format];
    imwrite(fr, file_name);
end
flag = 0;