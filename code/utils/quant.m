function quantized = quant(raw,thresh)

quantized = zeros(size(raw));
for t = 1:length(raw)
    quantized(t) = find(raw(t)>thresh,1,'last')-1;
end

end