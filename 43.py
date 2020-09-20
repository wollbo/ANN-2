# main
from core.functions import *
import pandas as pd
import sys


votes = pd.read_csv("data/votes.dat", header=None).to_numpy()
votes = np.reshape(votes, (349, 31))

party = pd.read_csv("data/mpparty.dat", header=None).to_numpy()  # Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
gender = pd.read_csv("data/mpsex.dat", header=None).to_numpy()  # Coding: Male 0, Female 1
district = pd.read_csv("data/mpdistrict.dat", header=None).to_numpy()

features = votes

rbf_units = 10
rbf_means = np.random.rand(10, 10)
weight_matrix = train_som2(features, rbf_means, n_iterations=100, max_d=4, eta=0.2)
# create mask to pick the relevant columns.
n_parties = np.unique(party)
n_genders = np.unique(gender)
n_districts = np.unique(district)

party_names = ["No party", "M", "Fp", "S", "V", "Mp", "Kd", "C"]
genders = ["Male", "Female"]

axes = []
fig = plt.figure()

for d in n_districts:
    res_matrix = reduce_sum_square(weight_matrix[:, district[district == d]], 0, 1)
    data = res_matrix.reshape(10, 10)
    axes.append(fig.add_subplot(5, 6, d + 1))
    axes[-1].set_title(f'{d}')
    plt.imshow(data)
    plt.axis('off')
plt.show()

axes = []
fig = plt.figure()


for n in n_parties:
    res_matrix = reduce_sum_square(weight_matrix[:, party[party == n]], 0, 1)
    data = res_matrix.reshape(10, 10)
    axes.append(fig.add_subplot(2, 4, n + 1))
    axes[-1].set_title(party_names[n])
    plt.imshow(data)
    plt.axis('off')
fig.tight_layout()
plt.show()

axes = []
fig = plt.figure()


for g in n_genders:
    res_matrix = reduce_sum_square(weight_matrix[:, gender[gender == g]], 0, 1)
    data = res_matrix.reshape(10, 10)
    axes.append(fig.add_subplot(1, 2, g + 1))
    axes[-1].set_title(genders[g])
    plt.imshow(data)
    plt.axis('off')
fig.tight_layout()
plt.show()




