import math

import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Tuple

import torch
from torch.nn.functional import softmax

from .target_input_distribution import TargetInputDistribution
from .epig import epig_from_logprobs, epig_from_probs

class AcquisitionFunction:
    def __init__(self, name: str, query_n_points: Optional[int] = None):
        self.name           = name
        self.query_n_points = query_n_points

    def __call__(self, value: Union[int, float]):
        return None
    
    def order_acq_scores(self, acq_scores: np.ndarray, return_sorted: bool):
        
        if return_sorted:
            pairs       = list(zip(*sorted(zip(acq_scores, np.arange(len(acq_scores))), key=lambda x: x[0], reverse=True)))
        else:
            pairs       = [acq_scores, np.arange(len(acq_scores))]

        # return N maximum Variation Ratios
        max_acq_scores  = np.array(pairs[0][:self.query_n_points])
        query_idxs      = np.array(pairs[1][:self.query_n_points])
        return max_acq_scores, query_idxs
        
class Random(AcquisitionFunction):

    def __init__(self, query_n_points):
        super().__init__(name='Random', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[Optional[np.ndarray], np.ndarray]:
        # pool_probs = kwargs['model'].predict_proba(Xpool)

        # Select randomly from pool according to uniform distribution
        n_samples = self.query_n_points if self.query_n_points is not None else len(Xpool)
        return np.ones(n_samples), np.random.randint(0, len(Xpool), size=n_samples) 

class VariationRatios(AcquisitionFunction):

    def __init__(self, query_n_points):
        super().__init__(name='VariationRatios', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pool_probs = kwargs['model'].predict_proba(Xpool)

        # Compute Variation Ratios and sort
        acq_scores      = 1 - pool_probs.max(axis=1)
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

class MinimumMargin(AcquisitionFunction):

    def __init__(self, query_n_points):
        super().__init__(name='MinimumMargin', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pool_probs = kwargs['model'].predict_proba(Xpool)
        
        # Compute Minimum margin (by maximizing reverse) and sort
        acq_scores      = 1 - (np.partition(pool_probs, -2, axis=1)[:, -1] - np.partition(pool_probs, -2, axis=1)[:, -2])
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)
    
class Entropy(AcquisitionFunction):

    def __init__(self, query_n_points):
        super().__init__(name='Entropy', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pool_probs = kwargs['model'].predict_proba(Xpool)
        
        # Compute Entropy and sort
        acq_scores      = - sum([pool_probs[:, cat] * np.log(pool_probs[:, cat]) for cat in range(pool_probs.shape[1])])
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

class BALD(AcquisitionFunction):

    def __init__(self, query_n_points, n_posterior_samples: int = 1000, seed: int = 0):
        # Set class-wide sampling parameters
        self.n_posterior_samples    = n_posterior_samples
        self.seed                   = seed

        super().__init__(name='BALD', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if kwargs['model'].__class__.__name__ in ['BinaryGenerativeGaussian', 'BayesianLogisticRegression', 'GaussianProcessClassifier']:
            # Returns log-probs
            posterior_samples   = kwargs['model'].sample(Xpool, n_samples=self.n_posterior_samples)
            pool_probs_sample   = torch.FloatTensor(posterior_samples)
            log_probs_sample    = torch.log(pool_probs_sample)
        else:
            # Returns logits
            posterior_samples   = kwargs['model'].sample(Xpool, n_samples=self.n_posterior_samples, seed=self.seed)
            log_probs_sample    = (posterior_samples - torch.logsumexp(posterior_samples, dim=0, keepdim=True))
            pool_probs_sample   = torch.softmax(posterior_samples, dim=0)
        
        pool_probs, log_probs   = pool_probs_sample.mean(axis=1).T, log_probs_sample.mean(axis=1).T

        ### BALD score estimation as of Gal, et al.: https://arxiv.org/pdf/1703.02910.pdf 
        # Compute entropy term
        entropy_term        = - sum([pool_probs[:, cat] * log_probs[:, cat] for cat in range(pool_probs.shape[1])])

        # Sample the posterior and compute disagreement term
        disagreement_term   = (pool_probs_sample * log_probs_sample).sum(axis=0).mean(axis=0)
        
        # Compute final acq-scores
        acq_scores          = entropy_term + disagreement_term
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

class EPIG(AcquisitionFunction):

    def __init__(self, query_n_points, epig_type: str, target_input_distribution: TargetInputDistribution, n_posterior_samples: int = 1000, n_target_input_samples: int = 100, seed: int = 0):
        # Make target-input distribution accesible
        self.target_input_distribution = target_input_distribution

        # Set class-wide sampling parameters
        self.epig_type              = epig_type 
        self.n_posterior_samples    = n_posterior_samples
        self.n_target_input_samples = n_target_input_samples
        self.seed                   = seed

        super().__init__(name='EPIG', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        K, M        = self.n_posterior_samples, self.n_target_input_samples

        # Sample x* values from the target input distribution
        Xstar                       = self.target_input_distribution.sample(M, seed=self.seed)
        self.Xstar                  = Xstar

        # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior 
        posterior_pool_samples      = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
        posterior_target_samples    = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)
        
        if self.epig_type == 'logprobs':
            acq_scores = epig_from_logprobs(Xpool, K, posterior_pool_samples, posterior_target_samples, kwargs)
        elif self.epig_type == 'probs':
            acq_scores = epig_from_probs(Xpool, K, posterior_pool_samples, posterior_target_samples, kwargs)
        else:
            raise NotImplementedError(f'Unknown EPIG type: {self.epig_type}')

        assert torch.all((acq_scores + 1e-6 >= 0) & (acq_scores <= math.inf)).item(), "Acquisition scores are not valid!"

        # Sort values
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)