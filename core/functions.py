# main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import sys
# unit variance
# 6 units needed for 0.1
# 8 units for 0.01
# 11 units for 0.001


def gaussian_kernel(x, mu, sigma=1):
    return np.exp(-np.square(np.abs(x-mu))/(2*np.square(sigma)))


def generate_data(lower, upper, n_samples=None, step=0.1):
    n_samples = n_samples if n_samples else np.floor((upper-lower)/step).astype(int)
    data = np.linspace(lower, upper, num=n_samples, endpoint=True)
    return data.reshape(data.size, 1)


def generate_gmm(means, variances, n_samples):
    """Generates data from an equally weighted 1-D gaussian mixture model"""
    indices = np.random.choice(np.size(means), n_samples)
    data = np.array([np.random.normal(means[index], variances[index]) for index in indices])
    sorted = np.sort(data)
    return np.absolute(sorted.reshape(data.size, 1))


def add_noise(data, sigma):
    return data + np.random.normal(0, sigma, size=data.shape)


def delta_update(data, f_train, rbf_means, rbf_variance, weight_matrix, learning_rate, norm=0.01):
    rbf_vector = gaussian_kernel(data, rbf_means, rbf_variance)
    error = f_train-rbf_vector * weight_matrix
    return weight_matrix + 1/2 * learning_rate * error * rbf_vector


def read_dat(path, regex="\s+", names=None):
    return pd.read_csv(path, sep=regex, names=names) if names else pd.read_csv(path, sep=regex, header=None)


def cl_rbf(input_data, means, n_iterations, m_dist=1, eta=0.01, initial_plot=False):
    weight_matrix = np.random.normal(means, 0.01)  # initialize weight matrix
    if initial_plot:
        plt.scatter(weight_matrix[:, 0], weight_matrix[:, 1])
    for n in range(n_iterations):
        input_data = np.random.permutation(input_data)
        sigma_decay = np.exp(-(n+1))
        for data in input_data:
            win_idx = np.argmin(reduce_sum_square(data, weight_matrix))
            neighborhood = [reduce_sum_square(gaussian_kernel(weight_matrix[win_idx], weight, sigma=sigma_decay), np.zeros(gaussian_kernel(weight_matrix[win_idx], weight).shape), axis=0) for weight in weight_matrix]
            winners = [idx for idx, win in enumerate(neighborhood) if win > m_dist]
            for win_idx in winners:
                weight_matrix[win_idx, :] = weight_matrix[win_idx, :] + eta * (data-weight_matrix[win_idx, :])
    return weight_matrix


def train_som(data, means, n_iterations, max_n, eta=0.2): # Use Hamming distance to find winning node in weight matrix vs prop
    weight_matrix = np.random.uniform(size=(means.size, data.shape[1]))
    for n in range(n_iterations):
        eta = eta * 0.99
        indices = np.random.permutation(data.shape[0])
        n_indices = max(0, np.round(max_n-n*(max_n/n_iterations)))
        for idx in indices:
            win_idx = np.argmin(reduce_sum_square(data[idx, :], weight_matrix))
            winners = np.mod(np.arange(win_idx-np.floor(n_indices/2), win_idx+np.ceil(n_indices/2)+1), weight_matrix.shape[0]).astype(int)
            for win in winners:
                dist = 1+(abs(np.where(winners == win)[0]-np.where(winners == win_idx)[0]))
                weight_matrix[win, :] = weight_matrix[win, :] + eta * (data[idx, :] - weight_matrix[win, :])/dist
    return weight_matrix


def train_som2(data, means, n_iterations, max_d=4, decay=0.95, eta=0.2):
    weight_matrix = np.random.uniform(size=(means.size, data.shape[1]))
    for n in range(n_iterations):
        eta = eta * 0.99
        indices = np.random.permutation(data.shape[0])
        n_dist = max_d * np.exp(- n**2/decay)
        for idx in indices:
            dist_matrix = np.zeros(weight_matrix.shape[0])
            win_idx = np.argmin(reduce_sum_square(data[idx, :], weight_matrix))
            # dist matrix of manhattan distances to winner
            for w in range(weight_matrix.shape[0]):
                dist_matrix[w] = max(abs((np.mod(win_idx, 10) - np.mod(w, 10))), abs(((win_idx - np.mod(win_idx, 10)) / 10) - ((w - np.mod(w, 10)) / 10)))
            winners = np.where(dist_matrix < n_dist)
            for win in winners:
                weight_matrix[win, :] = weight_matrix[win, :] + eta * (data[idx, :] - weight_matrix[win, :])
    return weight_matrix


def reduce_sum_square(a, b, axis=1):
    return np.sum(np.square(a - b), axis=axis)
