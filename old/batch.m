% batch

step_size = 0.1;
x_train = linspace(0, 2*pi, 2*pi/step_size)';
x_test = linspace(0.05, 2*pi, 2*pi/step_size)';
N = length(x_train);
sigma = 1;

error_vec_sin = [];
error_vec_sign = [];

f_train_sin = sin(2*x_train);
f_train_sign = sign(f_train_sin);

f_test_sin = sin(2*x_test);
f_test_sign = sign(f_test_sin);


%f_train_sin = add_noise(sin(2*x_train), 0.1);
%f_train_sign = add_noise(sign(f_train_sin), 0.1);

%f_test_sin = add_noise(sin(2*x_test), 0.1);
%f_test_sign = add_noise(sign(f_test_sin), 0.1);

for rbf_units = 1:N
    
    f_estimate_sin = function_estimate(rbf_units, sigma, x_train, x_test, f_train_sin);
    f_estimate_sign = function_estimate(rbf_units, sigma, x_train, x_test, f_train_sign);
    
    error_vec_sin = [error_vec_sin mean(abs(f_estimate_sin-f_test_sin))];
    error_vec_sign = [error_vec_sign mean(abs(f_estimate_sign-f_test_sign))];
    
end

tresholds = [0.1 0.01 0.001];
[t_sin_vec, t_sin_tresholds] = treshold_run(tresholds, error_vec_sin);
[t_sign_vec, t_sign_tresholds] = treshold_run(tresholds, error_vec_sign);


%%
figure(2)
hold on
for i = 1:length(tresholds)
    plot(1:N, function_estimate(t_sin_tresholds(i), sigma, x_train, x_test, f_train_sin))
    
end
plot(1:N, f_test_sin)
legend('RBF units =' + string(t_sin_tresholds(1)), 'RBF units =' + string(t_sin_tresholds(2)), 'RBF units =' + string(t_sin_tresholds(3)), 'True function')
hold off

%figure(1)
%plot(1:N, error_vec_sin, 1:N, t_sin, '*')
%hold on
%plot(1:N, error_vec_sign, 1:N, t_sign, '*')
%hold off


