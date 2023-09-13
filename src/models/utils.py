import numpy as np
from scipy.linalg import solve

import pickle
from collections import defaultdict
from typing import List

# Define wrapper function for sampling the GP Classifier
def GP_sample(self, X, n_samples, seed=0, verbose=False):
    ### ADJUSTED THE SKLEARN IMPLEMENTATION: https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/gaussian_process/_gpc.py#L487 ###
    ### They refer to Algorithm 3.2 of this paper: https://gaussianprocess.org/gpml/chapters/RW.pdf ###
    np.random.seed(seed)
    
    # Define sigmoid function
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    
    # Get estimators from model
    num_classes     = self.classes_.__len__()
    estimators_     = [self.base_estimator_] if num_classes == 2 else self.base_estimator_.estimators_

    output_probs    = []
    for base in estimators_:

        K_star          = base.kernel_(base.X_train_, X)  # K_star =k(x_star)
        f_star          = K_star.T.dot(base.y_train_ - base.pi_)  # Line 4
        v               = solve(base.L_, base.W_sr_[:, np.newaxis] * K_star)  # Line 5
        var_f_star      = base.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)

        covar_f_star    = np.eye(len(var_f_star)) * var_f_star

        z               = np.random.multivariate_normal(f_star, covar_f_star, size=n_samples)
        output_probs.append(sigmoid(z))

    output_probs = np.array(output_probs)
    if num_classes == 2:
        output_probs = np.vstack([1-output_probs, output_probs])
        
    if verbose:
        numerator   = output_probs.mean(axis=1).T
        denominator = numerator.sum(axis=1)

        print(f"Sample mean:\n {numerator / denominator[:, None]}\n")
        print(f"Model mean:\n {self.predict_proba(X)}")
        
        assert np.allclose(self.predict_proba(X), numerator / denominator[:, None], atol=1e-2)
    
    return output_probs

def combine_results(acq_functions: List, save_dir: str = "../reports/2D_toy", seeds=[0]):
    train_results, test_results = defaultdict(dict), defaultdict(dict)
    for i, acq_fun in enumerate(acq_functions):
        for experiment in ['binary', 'multiclass']:
                            
            for j, seed in enumerate(seeds):

                with open(f'{save_dir}/{experiment}/{acq_fun}/performance_seed{seed}.pkl', 'rb') as f:
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