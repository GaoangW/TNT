function E = estEntropy(p)

N = length(p);
if sum(p(p==Inf))~=0
    E = 0;
    return
end
if sum(p)~=0
    p = p/sum(p);
end
E = 0;
for n = 1:N
    if p(n)~=0           
        E = E-p(n)*log2(p(n));                
    end
end