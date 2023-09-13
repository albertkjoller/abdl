import numpy as np
from typing import Optional, Union, Tuple

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
        pool_probs = kwargs['model'].predict_proba(Xpool)

        # Select randomly from pool according to uniform distribution
        n_samples = self.query_n_points if self.query_n_points is not None else len(pool_probs)
        return np.ones(n_samples), np.random.randint(0, len(pool_probs), size=n_samples) 

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
        
        # Compute Variation Ratios and sort
        acq_scores      = - sum([pool_probs[:, cat] * np.log(pool_probs[:, cat]) for cat in range(pool_probs.shape[1])])
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

class BALD(AcquisitionFunction):

    def __init__(self, query_n_points, n_samples: int = 1000, seed: int = 0):
        self.n_samples  = n_samples
        self.seed       = seed

        super().__init__(name='BALD', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pool_probs          = kwargs['model'].predict_proba(Xpool)
        
        ### BALD score estimation as of Gal, et al.: https://arxiv.org/pdf/1703.02910.pdf 
        # Compute entropy term
        entropy_term        = - sum([pool_probs[:, cat] * np.log(pool_probs[:, cat]) for cat in range(pool_probs.shape[1])])
        # Sample the posterior and compute disagreement term
        posterior_samples   = kwargs['model'].sample(Xpool, n_samples=self.n_samples, seed=self.seed)
        disagreement_term   = (posterior_samples * np.log(posterior_samples + 1e-9)).sum(axis=0).mean(axis=0)
        
        # Compute final acq-scores
        acq_scores          = entropy_term + disagreement_term
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

class EPIG(AcquisitionFunction):

    def __init__(self, query_n_points):
        super().__init__(name='EPIG', query_n_points=query_n_points)

    def __call__(self, value: Union[int, float]):
        return value
