# main
from core.functions import *
import pandas as pd
import sys



square = True
batch = False
epochs = 100
learning_rate = 0.1

step_size = 0.1
sigma = 0.1

features = pd.read_csv("data/animals.dat", header=None).to_numpy()
features = np.reshape(features, (32, 84))

labels = read_dat("data/animalnames.txt", names=None)[0]

st_dev = 0.1
rbf_units = 100

rbf_means = np.linspace(0, 1, rbf_units)

rbf_variance = st_dev * np.ones(rbf_means.shape)
weight_matrix = train_som(features, rbf_means, n_iterations=20, max_n=50)

indices = [np.argmin(np.sum(np.abs(data-weight_matrix), axis=1)) for data in features]
tmp_dict = labels.to_dict()
new_dict = {indices[key]: value for key, value in tmp_dict.items()}


sorted = pd.Series(dict(sorted(new_dict.items())))
#print(len(sorted))

print(sorted)
