function [f_estimate, absolute_error] = function_estimate(rbf_units, sigma, x_train, x_test, func)

    rbf_means = linspace(0, 2*pi, rbf_units)';
    rbf_variance = sigma*ones(rbf_units,1); %randn(rbf_units, 1);

    w = create_rbf_weights(x_train, func, rbf_means, rbf_variance);
    rbf_matrix = create_rbf_matrix(x_test, rbf_means, rbf_variance);
    f_estimate = rbf_matrix * w;

    %plot(1:N, f_estimate, 1:N, sin(2*x_test))
    %plot(sin(2*x_test)-f_estimate)
    absolute_error = mean(abs(f_estimate-func));
end
