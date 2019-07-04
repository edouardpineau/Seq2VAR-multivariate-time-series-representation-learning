import numpy as np
import argparse


def permutation_TS(p, T, P, seed):

    """

    :param p: number of variables
    :param T: time length of the time series
    :param P: permutation matrix
    :param seed: random seed
    :return:
    """
    np.random.seed(seed)
    v = np.random.randn(p, T)
    for i in range(1, T):
        v[:, i] = np.matmul(P, v[:, i-1])
    return v


def generate_permutation_ts(p, T, n_class, n_samples_per_class, sd=0):

    """

    :param p: number of variables
    :param T: time length of the time series
    :param n_class: number of desired permutations matrix
    :param n_samples_per_class:  number of sample per class
    :param sd: standard deviation of added noise (default 0)
    :return:
    """
    
    seeds_W = np.random.choice(range(np.max([1000, n_class])),
                               n_class,
                               replace=False)
    samples, Ws, labels, initial_cond = [], [], [], []
    matrices_labels = {}
    for id_seed, sw in enumerate(seeds_W):
        p_ind = list(range(p))
        np.random.shuffle(p_ind)
        while np.sum([int(p_ind[i] == i) for i in range(p)]) > 0:
            np.random.shuffle(p_ind)
        P = np.eye(p)[np.ix_(range(p), p_ind)]
        matrices_labels[id_seed] = P
        seeds_sample = np.random.choice(range(np.max([1000, n_class])),
                                        n_samples_per_class,
                                        replace=False)

        for seed in seeds_sample:
            X = permutation_TS(p, T, P, seed)
            samples.append(X)
            labels.append(id_seed)
            
    samples = np.stack(samples)
    labels = np.stack(labels)

    samples[:, :, 1:] = samples[:, :, 1:] + np.random.randn(*samples[:, :, 1:].shape)*sd
    
    return np.expand_dims(samples, -1), matrices_labels, labels