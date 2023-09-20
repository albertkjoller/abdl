import numpy as np
from sklearn.datasets import make_moons, make_classification

def generate_sine(N_initial, N_test, N_pool, seed=42):
    np.random.seed(42)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    xaxis   = np.random.permutation(np.arange(0, 2*np.pi, 6 / (N_initial + N_test + N_pool)))
    labels  = (sigmoid(np.sin(2*np.pi*xaxis / 4)) > 0.5).astype(int) # threshold a sigmoid

    Xtrain, Xtest, Xpool  = xaxis[:N_initial], xaxis[N_initial:N_initial+N_test], xaxis[N_initial+N_initial+N_test:]
    ytrain, ytest, ypool  = labels[:N_initial], labels[N_initial:N_initial+N_test], labels[N_initial+N_initial+N_test:]
    return Xtrain.reshape(-1,1), ytrain, Xtest.reshape(-1,1), ytest, Xpool.reshape(-1,1), ypool, np.arange(0, 2*np.pi, 6 / (N_initial + N_test + N_pool))

def generate_moons(N_initial_per_class, N_test, N_pool, noise=0.075):
    # Get total size of dataset
    N = N_initial_per_class + N_test + N_pool

    # Generate moons dataset
    X, y = make_moons(N, noise=noise, random_state=0)
    
    # Partition dataset
    Xtrain, ytrain  = np.vstack([X[y == 0][:N_initial_per_class], X[y == 1][:N_initial_per_class]]), np.hstack([y[y == 0][:N_initial_per_class], y[y == 1][:N_initial_per_class]])
    train_idxs      = np.hstack([np.arange(len(y))[y == 0][:N_initial_per_class], np.arange(len(y))[y == 1][:N_initial_per_class]])

    order           = np.random.permutation(np.setdiff1d(np.arange(N), train_idxs))
    Xtest,  ytest   = X[order[N_initial_per_class:N_test+N_initial_per_class]], y[order[N_initial_per_class:N_test+N_initial_per_class]]
    Xpool,  ypool   = X[order[N_initial_per_class+N_test:]], y[order[N_initial_per_class+N_test:]]

    return Xtrain, ytrain, Xtest, ytest, Xpool, ypool

def generate_multiclass(N_initial_per_class, N_test, N_pool, num_classes=4, noise=0.075):
    # Get total size of dataset
    N = N_initial_per_class + N_test + N_pool

    # Generate moons dataset
    X, y = make_classification(N, n_classes=num_classes, n_clusters_per_class=1, n_informative=2, n_features=2, n_redundant=0, scale=noise, random_state=0)

    # Partition dataset
    Xtrain, ytrain  = np.vstack([X[y == i][:N_initial_per_class] for i in range(num_classes)]), np.hstack([y[y == i][:N_initial_per_class] for i in range(num_classes)])
    train_idxs      = np.hstack([np.arange(len(y))[y == i][:N_initial_per_class] for i in range(num_classes)])
    
    order           = np.random.permutation(np.setdiff1d(np.arange(N), train_idxs))
    Xtest,  ytest   = X[order[N_initial_per_class:N_test+N_initial_per_class]], y[order[N_initial_per_class:N_test+N_initial_per_class]]
    Xpool,  ypool   = X[order[N_initial_per_class+N_test:]], y[order[N_initial_per_class+N_test:]]

    return Xtrain, ytrain, Xtest, ytest, Xpool, ypool