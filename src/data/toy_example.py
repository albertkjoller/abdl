import numpy as np
from sklearn.datasets import make_moons

def generate_data(N_initial, N_test, N_pool, noise=0.075):
    # Get total size of dataset
    N = N_initial + N_test + N_pool

    # Generate moons dataset
    X, y = make_moons(N, noise=noise)

    # Partition dataset
    order           = np.random.permutation(np.arange(N))
    Xtrain, ytrain  = X[order[:N_initial]], y[order[:N_initial]]
    Xtest,  ytest   = X[order[N_initial:N_test+N_initial]], y[order[N_initial:N_test+N_initial]]
    Xpool,  ypool   = X[order[N_initial+N_test:]], y[order[N_initial+N_test:]]

    return Xtrain, ytrain, Xtest, ytest, Xpool, ypool