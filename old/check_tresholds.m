function treshold_indices = check_tresholds(vector, tresholds)

for i = 1:length(tresholds)
    [~, treshold_indices(i)] = min(abs(vector(vector>tresholds(i))-tresholds(i)));
end

        