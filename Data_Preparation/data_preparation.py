import pickle
import numpy as np
from numpy import load as np_load


def Data_Prepar_1():
    """Load & preprocess the SIMULATED MCG data; return full X, y arrays."""
    noisy = np_load('Dataset/Real_Data/noise_1280.npy', allow_pickle=True)
    clean = np_load('Dataset/Real_Data/label_1280.npy', allow_pickle=True)

    noisy = noisy / 200
    clean = clean / 200

    # zero-center each sample
    noisy = noisy - noisy.mean(axis=1, keepdims=True)
    clean = clean - clean.mean(axis=1, keepdims=True)

    # filter out extreme mismatches
    diff_max = np.max(np.abs(noisy - clean), axis=1)
    mask     = diff_max < 1
    noisy    = noisy[mask]
    clean    = clean[mask]

    X = np.expand_dims(noisy, axis=2)
    y = np.expand_dims(clean, axis=2)
    return X, y


def Data_Prepar_2():
    """Load & preprocess the SIMULATED MCG data; return full X, y arrays."""
    noisy = np_load('Dataset/Simulated_Data/noise_1280.npy', allow_pickle=True)
    clean = np_load('Dataset/Simulated_Data/label_1280.npy', allow_pickle=True)

    noisy = noisy / 200
    clean = clean / 200

    # filter out extreme mismatches
    diff_max = np.max(np.abs(noisy - clean), axis=1)
    mask     = diff_max < 0.525
    noisy    = noisy[mask]
    clean    = clean[mask]

    X = np.expand_dims(noisy, axis=2)
    y = np.expand_dims(clean, axis=2)
    return X, y
