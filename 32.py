# main
from core.functions import *

# 0.5 variance sinus
# 8 units needed for 0.1
# 13 units for 0.01
# 20 units for 0.001

# 14 units for square sign
# 62, 0.06 best for square uniform
#

# Use gaussian function to place network weights?


square = True
gmm = False
batch = True # Online: Decreased variance, a lot worse in the mean! sensitive!
epochs = 100
learning_rate = 0.01

st_dev = 0.08
rbf_units = 50
step_size = 0.1
sigma = 0.0

x_train = generate_data(0, 2*np.pi, step=step_size)
x_test = generate_data(0.05, 2*np.pi, step=step_size)

f_test = np.sin(2*x_test)
f_train = np.sin(2*x_train)

if square:
    f_test = np.sign(f_test)
    f_train = np.sign(f_train)
    if gmm:
        means = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        variances = np.array([0.6, 0.6, 0.6, 0.6, 0.6])

if sigma:
    f_long = np.vstack((f_train, f_test))
    f_train = np.split(add_noise(f_long, sigma), 2)[0]
    f_test = np.split(add_noise(f_long, sigma), 2)[1]

N = x_train.size

rbf_means = generate_gmm(means, variances, rbf_units) if gmm else generate_data(0, 2*np.pi, n_samples=rbf_units)
rbf_variance = st_dev * np.ones(rbf_means.shape)


if batch:
    rbf_matrix = gaussian_kernel(x_train, np.transpose(rbf_means), np.transpose(rbf_variance))
    gram = np.transpose(rbf_matrix) @ rbf_matrix
    weight_matrix = np.linalg.inv(gram + 1e-6 * np.eye(gram.shape[0])) @ np.transpose(rbf_matrix) @ f_train
    f_estimate = gaussian_kernel(x_test, np.transpose(rbf_means), np.transpose(rbf_variance)) @ weight_matrix

else:
    weight_matrix = np.random.normal(0, 0.1, rbf_variance.shape)
    for e in range(epochs):
        t_indices = np.random.permutation(len(x_train))
        for index in t_indices:
            weight_matrix = delta_update(x_train[index], f_train[index], rbf_means, rbf_variance, weight_matrix, learning_rate)
    f_estimate = gaussian_kernel(x_test, np.transpose(rbf_means), np.transpose(rbf_variance)) @ weight_matrix
error = np.mean(np.abs(f_estimate-f_test))

print(error)
plt.plot(f_test)
plt.plot(f_estimate)

plt.legend(('true', 'estimated'))
plt.show()
