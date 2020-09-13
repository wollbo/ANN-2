function weights = delta_update(x_train, func, rbf_means, rbf_variances, weights, learning_rate)

rbf_vector = zeros(size(rbf_means));
% calculate rbf_vector for each sample
for j = 1:length(rbf_means)
        rbf_vector(j) = gaussian_kernel(x_train, rbf_means(j), rbf_variances(j));
end
error = func-rbf_vector'*weights;
weights = weights + learning_rate * (error) * rbf_vector;

    

end

