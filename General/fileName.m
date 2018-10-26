function file_name = fileName(idx, num_digit)

file_name = [];
for n = 1:num_digit
    num = 10^n;
    r = idx/num;
    if r>=0.1 && r<1
        for m = 1:num_digit-n
            file_name = [file_name,'0'];
        end
        file_name = [file_name, num2str(idx)];
    end
end