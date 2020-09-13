function rbf_matrix = create_rbf_matrix(x, mu, sigma)

for i = 1:length(x)
    for j = 1:length(mu)
        rbf_matrix(i,j) = gaussian_kernel(x(i), mu(j), sigma(j));
    end
end
end