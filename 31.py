
from core.functions import *

# unit variance sinus
# 6 units needed for 0.1
# 8 units for 0.01
# 11 units for 0.001

# 14 units for square sign

step_size = 0.1

x_train = generate_data(0, 2*np.pi, step=step_size)
x_test = generate_data(0.05, 2*np.pi, step=step_size)

N = x_train.size
st_dev = 1
rbf_units = 11
rbf_means = generate_data(0, 2*np.pi, n_samples=rbf_units)
rbf_variance = st_dev * np.ones(rbf_means.shape)

weight_matrix = np.random.normal(0, 1, rbf_variance.shape)
rbf_matrix = gaussian_kernel(x_train, np.transpose(rbf_means), np.transpose(rbf_variance))
gram = np.transpose(rbf_matrix) @ rbf_matrix

f_train = np.sin(2*x_train)
#f_train = np.sign(f_train)
weight_matrix = np.linalg.inv(gram) @ np.transpose(rbf_matrix) @ f_train
f_estimate = gaussian_kernel(x_test, np.transpose(rbf_means), np.transpose(rbf_variance)) @ weight_matrix
#f_estimate = np.sign(f_estimate)
f_test = np.sin(2*x_test)
#f_test = np.sign(f_test)


error = np.mean(np.abs(f_estimate-f_test))

print(error)
plt.plot(f_test)
plt.show()
