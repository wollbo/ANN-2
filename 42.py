# main
from core.functions import *
import pandas as pd
import sys


features = pd.read_csv("data/cities.dat", header=None).to_numpy()

st_dev = 0.1
rbf_units = 10
rbf_means = np.linspace(0, 1, rbf_units)
weight_matrix = train_som(features, rbf_means, n_iterations=100, max_n=2, eta=0.2)
indices = {idx: np.argmin(reduce_sum_square(data, weight_matrix, axis=1)) for idx, data in enumerate(features)}
indices = dict(sorted(indices.items(), key=lambda x: x[1]))


plot_listx = [features[key, 0] for key, value in indices.items()]
plot_listx.append(plot_listx[0])

plot_listy = [features[key, 1] for key, value in indices.items()]
plot_listy.append(plot_listy[0])

plt.plot(plot_listx, plot_listy, 'xr-')
plt.scatter(weight_matrix[:, 0], weight_matrix[:, 1])
plt.scatter(features[:, 0], features[:, 1])


plt.legend(["Path",  "SOM-unit", "City location"])
plt.show()

