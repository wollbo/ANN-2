function [treshold_vector, treshold_indices] = treshold_run(tresholds, error_vec)
%TRESHOLD_RUN Summary of this function goes here

treshold_indices = check_tresholds(error_vec, tresholds);
treshold_vector = nan(size(error_vec));
treshold_vector(treshold_indices) = error_vec(treshold_indices);

end

