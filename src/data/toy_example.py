import numpy as np
from sklearn.datasets import make_moons, make_classification, make_blobs

def generate_sine(N_initial, N_test, N_pool, seed=42):
    np.random.seed(42)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    xaxis   = np.random.permutation(np.arange(0, 2*np.pi, 6 / (N_initial + N_test + N_pool)))
    labels  = (sigmoid(np.sin(2*np.pi*xaxis / 4)) > 0.5).astype(int) # threshold a sigmoid

    Xtrain, Xtest, Xpool  = xaxis[:N_initial], xaxis[N_initial:N_initial+N_test], xaxis[N_initial+N_initial+N_test:]
    ytrain, ytest, ypool  = labels[:N_initial], labels[N_initial:N_initial+N_test], labels[N_initial+N_initial+N_test:]
    return Xtrain.reshape(-1,1), ytrain, Xtest.reshape(-1,1), ytest, Xpool.reshape(-1,1), ypool, np.arange(0, 2*np.pi, 6 / (N_initial + N_test + N_pool))

def generate_moons(N_initial_per_class, N_test, N_pool, noise=0.2, seed=42):

    # Generate initial points
    Xtrain, ytrain = make_moons(N_initial_per_class * 2, noise=noise, random_state=seed)
    Xtrain, ytrain = np.vstack([Xtrain[ytrain == 0][:N_initial_per_class], Xtrain[ytrain == 1][:N_initial_per_class]]), np.hstack([ytrain[ytrain == 0][:N_initial_per_class], ytrain[ytrain == 1][:N_initial_per_class]])

    # Generate moons dataset
    X_, y_ = make_moons(N_test + N_pool, noise=noise, random_state=seed)
    
    # Generate pool and test sets
    Xtest,  ytest   = X_[:N_test], y_[:N_test]
    Xpool,  ypool   = X_[N_test:], y_[N_test:]

    return Xtrain, ytrain, Xtest, ytest, Xpool, ypool

def generate_multiclass(N_initial_per_class, N_test, N_pool, num_classes=4, noise=0.35, seed=42):

    # Generate initial points
    Xtrain, ytrain = make_blobs(n_samples=N_initial_per_class * num_classes, centers=num_classes, cluster_std=noise, center_box=(-3, 3), random_state=seed) 
    Xtrain, ytrain  = np.vstack([Xtrain[ytrain == i][:N_initial_per_class] for i in range(num_classes)]), np.hstack([ytrain[ytrain == i][:N_initial_per_class] for i in range(num_classes)])

    # Generate moons dataset
    X_, y_ = make_blobs(n_samples=N_test + N_pool, centers=num_classes, cluster_std=noise, center_box=(-3, 3), random_state=seed) 
    
    # Generate pool and test sets
    Xtest,  ytest   = X_[:N_test], y_[:N_test]
    Xpool,  ypool   = X_[N_test:], y_[N_test:]

    return Xtrain, ytrain, Xtest, ytest, Xpool, ypool