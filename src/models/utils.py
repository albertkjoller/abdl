import numpy as np
from scipy.linalg import solve

import pickle
from collections import defaultdict
from typing import List

# Define wrapper function for sampling the GP Classifier
def GP_sample(self, X, n_samples, seed=0, verbose=False):
    ### ADJUSTED THE SKLEARN IMPLEMENTATION: https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/gaussian_process/_gpc.py#L487 ###
    ### FOR MULTI-CLASS CLASSIFICATION, THIS FUNCTION RELIES ON SKLEARN'S OneVsRestClassifier IMPLEMENTATION (here, copied)
    ### They refer to Algorithm 3.2 of this paper: https://gaussianprocess.org/gpml/chapters/RW.pdf ###
    np.random.seed(seed)
    
    # Define sigmoid function
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    rev_sigmoid = lambda x: np.log(x / (1 - x))

    # Set dataset shape    
    num_points      = X.__len__()

    # Get estimators from model
    num_classes     = self.classes_.__len__()
    estimators_     = [self.base_estimator_] if num_classes == 2 else self.base_estimator_.estimators_

    Y               = np.zeros((estimators_.__len__(), n_samples, num_points))
    # Iterate estimators in case of multi-class problem
    for i, base in enumerate(estimators_):
        # Compute kernel values, mean (f*), and variance (sigma(f*)) for the test points
        K_star          = base.kernel_(base.X_train_, X)  # K_star =k(x_star)
        f_star          = K_star.T.dot(base.y_train_ - base.pi_)  # Line 4
        v               = solve(base.L_, base.W_sr_[:, np.newaxis] * K_star)  # Line 5
        var_f_star      = base.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)

        # Sample latent function value, z ~ N(f_star, sqrt(var_f_star))
        z               = np.random.normal(f_star, np.sqrt(var_f_star), size=(n_samples, num_points))
        # Pass sample through sigmoid (assuming OneVsRestClassifier)
        Y[i, :]         = sigmoid(z)

    if num_classes == 2:
        return np.vstack([1-Y, Y]) 
        
    if verbose:
        print(f"Sample mean:\n {Y.mean(axis=1).T}\n\nModel mean:\n {self.predict_proba(X)}")
        assert np.allclose(self.predict_proba(X), Y.mean(axis=1).T, atol=1e-2)
    
    return Y / np.sum(Y, axis=0)


def combine_results(acq_functions: List, experiments: List[str], save_dir: str = "../reports/2D_toy", seeds=[0]):
    train_results, test_results = defaultdict(dict), defaultdict(dict)
    for i, acq_fun in enumerate(acq_functions):           
        for experiment in experiments:
            for j, seed in enumerate(seeds):

                with open(f'{save_dir}/{experiment}/{acq_fun}/seed{seed}/performance.pkl', 'rb') as f:
                    res         = pickle.load(f)
                    train_res   = np.array(res['train']).reshape(1, -1) if j == 0 else np.vstack([train_res, res['train']])
                    test_res    = np.array(res['test']).reshape(1, -1) if j == 0 else np.vstack([test_res, res['test']])
            
            if i == 0:
                train_results[experiment]['N_points']   = res['N_points']
                test_results[experiment]['N_points']    = res['N_points']

            train_results[experiment][acq_fun]          = np.mean(train_res, axis=0)
            test_results[experiment][acq_fun]           = np.mean(test_res, axis=0) 
            train_results[experiment][f'{acq_fun}_std'] = np.std(train_res, axis=0) / np.sqrt(train_res.shape[0])
            test_results[experiment][f'{acq_fun}_std']  = np.std(test_res, axis=0) / np.sqrt(train_res.shape[0])

    return train_results, test_results