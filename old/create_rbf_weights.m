function rbf_weights = create_rbf_weights(x, f, rbf_means, rbf_variance)
    rbf_matrix = create_rbf_matrix(x, rbf_means, rbf_variance);
    rbf_weights = (rbf_matrix'*rbf_matrix)\rbf_matrix'*f;
end