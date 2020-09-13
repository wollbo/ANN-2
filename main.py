# main
import numpy as np


def gaussian_kernel(x_1, mu, sigma):
    return np.exp(-np.square(np.abs(x_1-mu))/(2*np.square(sigma)))

step_size = 0.1

x_train = np.linspace(0, 2*np.pi, num=(2*np.pi/step_size), endpoint=True)
x_test = np.linspace(0.05, 2*np.pi, num=(2*np.pi/step_size), endpoint=True)
x_train = x_train.reshape(x_train.size,1)
x_test = x_test.reshape(x_test.size,1)

N = x_train.size
rbf_units = 25
rbf_means = np.linspace(0, 2*np.pi, rbf_units, endpoint=True)
rbf_means = rbf_means.reshape(rbf_means.size, 1)
rbf_variance = np.ones(rbf_means.shape)

weight_matrix = np.random.normal(0, 1, rbf_variance.shape)
rbf_matrix = gaussian_kernel(x_train, np.transpose(rbf_means), np.transpose(rbf_variance))
f_train = np.sin(2*x_train)

gram = np.transpose(rbf_matrix) @ rbf_matrix

weight_matrix = np.linalg.inv(gram) @ np.transpose(rbf_matrix) @ f_train

f_estimate = gaussian_kernel(x_test, np.transpose(rbf_means), np.transpose(rbf_variance)) @ weight_matrix

#print(f_estimate)
#print(gaussian_kernel(x_test, np.transpose(rbf_means), np.transpose(rbf_variance)))
print(np.mean(np.abs(f_train-f_estimate)))