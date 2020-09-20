# main
from core.functions import *
import sys

# unit variance sinus
# 6 units needed for 0.1
# 8 units for 0.01
# 11 units for 0.001

# 14 units for square sign

# Use gaussian function to place network weights

# 0.5 stdev, 20 units perfect reconstruction for batch mode!?!?!

square = True
batch = False
epochs = 100
learning_rate = 0.1

step_size = 0.1
sigma = 0.1

names=["input1", "input2", "function1", "function2"]

train = read_dat("data/ballist.dat", names).to_numpy()
x_train = train[:, 0:2]
f_train = train[:, 2:]
test = read_dat("data/balltest.dat", names).to_numpy()
x_test = test[:, 0:2]
f_test = test[:, 2:]

N = x_train.size
st_dev = 0.1
rbf_units = 20

rbf_means = np.array([[x, y] for x in np.linspace(min(f_train[:, 0]), max(f_train[:, 0]), 2) for y in np.linspace(min(f_train[:, 1]), max(f_train[:, 1]), 2)])
rbf_variance = st_dev * np.ones(rbf_means.shape)
weight_matrix = cl_rbf(x_train, rbf_means, n_iterations=100, m_dist=1.6, initial_plot=True)
plt.scatter(weight_matrix[:, 0], weight_matrix[:, 1])
plt.scatter(x_train[:,0], x_train[:,1])
plt.legend(["Initial placement", "RBF-units", "training data"])
plt.show()

sys.exit(0)

for e in range(epochs):
    t_indices = np.random.permutation(len(x_train))
    for index in t_indices:
        weight_matrix = delta_update(x_train[index], f_train[index], rbf_means, rbf_variance, weight_matrix, learning_rate)
f_estimate = gaussian_kernel(x_test, np.transpose(rbf_means), np.transpose(rbf_variance)) @ weight_matrix

error = np.mean(np.abs(f_estimate-f_test))

print(error)
plt.plot(f_test)
plt.plot(f_estimate)
plt.show()
