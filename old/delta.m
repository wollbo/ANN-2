% delta 

step_size = 0.1;
x_train = linspace(0, 2*pi, 2*pi/step_size)';
x_test = linspace(0.05, 2*pi, 2*pi/step_size)';
N = length(x_train);
sigma = 1;

f_train_sin = add_noise(sin(2*x_train), 0.1);
f_train_sign = add_noise(sign(f_train_sin), 0.1);

f_test_sin = add_noise(sin(2*x_test), 0.1);
f_test_sign = add_noise(sign(f_test_sin), 0.1);

learning_rate = 0.01;
epochs = 100;

error_vec_sin = [];
error_vec_sign = [];

for rbf_units = 1:N

error_vec_sin = [];
error_vec_sign = [];

weights_sin = randn(rbf_units,1);
weights_sign = weights_sin;

rbf_means = linspace(0, 2*pi, rbf_units)'; % do gaussn
rbf_variances = sigma*ones(rbf_units,1); %randn(rbf_units, 1);

for l = 1:epochs
rand_indices = randperm(length(x_train));

for i = 1:length(x_train)
    weights_sin = delta_update(x_train(rand_indices(i)), f_train_sin(rand_indices(i)), rbf_means, rbf_variances, weights_sin, learning_rate);
    weights_sign = delta_update(x_train(rand_indices(i)), f_train_sign(rand_indices(i)), rbf_means, rbf_variances, weights_sign, learning_rate);
end
end
rbf_matrix = create_rbf_matrix(x_test, rbf_means, rbf_variances);
error_sin = [error_sin mean((f_test_sin-rbf_matrix*weights_sin).^2)];
error_sign = [error_sign mean((f_test_sign-rbf_matrix*weights_sign).^2)];
end

%%