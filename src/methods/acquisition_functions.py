import numpy as np
import torch

from typing import Optional, Union, Tuple

from .target_input_distribution import TargetInputDistribution

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

    def __init__(self, query_n_points, n_posterior_samples: int = 1000, seed: int = 0):
        # Set class-wide sampling parameters
        self.n_posterior_samples    = n_posterior_samples
        self.seed                   = seed

        super().__init__(name='BALD', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pool_probs          = kwargs['model'].predict_proba(Xpool)
        
        ### BALD score estimation as of Gal, et al.: https://arxiv.org/pdf/1703.02910.pdf 
        # Compute entropy term
        entropy_term        = - sum([pool_probs[:, cat] * np.log(pool_probs[:, cat]) for cat in range(pool_probs.shape[1])])
        # Sample the posterior and compute disagreement term
        posterior_samples   = kwargs['model'].sample(Xpool, n_samples=self.n_posterior_samples, seed=self.seed)
        disagreement_term   = (posterior_samples * np.log(posterior_samples + 1e-9)).sum(axis=0).mean(axis=0)
        
        # Compute final acq-scores
        acq_scores          = entropy_term + disagreement_term
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

class EPIG(AcquisitionFunction):

    def __init__(self, query_n_points, target_input_distribution: TargetInputDistribution, n_posterior_samples: int = 1000, n_target_input_samples: int = 100, seed: int = 0, version: str = 'mine'):
        # Make target-input distribution accesible
        self.target_input_distribution = target_input_distribution

        # Set class-wide sampling parameters
        self.n_posterior_samples    = n_posterior_samples
        self.n_target_input_samples = n_target_input_samples
        self.seed                   = seed
        self.version                = version 

        super().__init__(name='EPIG', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        # Sample x* values from the target input distribution
        Xstar                   = self.target_input_distribution.sample(self.n_target_input_samples, seed=self.seed)
        
        if self.version == 'mine':
            # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior
            probs_pool  = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
            probs_targ  = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)

            # Define constants
            num_classes = probs_pool.shape[0]
            K, M        = self.n_posterior_samples, self.n_target_input_samples

            # Compute the joint term of the expression (summation on the numerator part of the fraction in the log)
            joint_term          = np.array([[(probs_pool[c, :, :] * probs_targ[c_star, :, j][:, None]).sum(axis=0) for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)])
            # Compute the independent term of the expression (summation on the denominator part of the fraction in the log)
            independent_term    = np.array([[probs_pool[c, :, :].sum(axis=0) * probs_targ[c_star, :, j].sum(axis=0) for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)])
            # Compute log-term
            log_term            = np.log(K * joint_term) - np.log(independent_term)
            # Wrap it up and compute the final acquisition scores
            acq_scores          = 1/M * (joint_term/K * log_term).sum(axis=(1, 0)) #.sum(axis=0)

            # Sort values
            return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

        else: # outdated - found from their repository - it is quite slow, compared to my implementation and yielded the same result
            ### TAKEN FROM THE AUTHORS OF THE EPIG PAPER: https://github.com/fbickfordsmith/epig/blob/main/src/uncertainty/bald.py 
            probs_pool  = torch.tensor(kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed))
            probs_targ  = torch.tensor(kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed))

            probs_pool = probs_pool.permute(1, 2, 0)  # [K, N_p, Cl]
            probs_targ = probs_targ.permute(1, 2, 0)  # [K, N_t, Cl]

            probs_pool              = probs_pool[:, :, None, :, None]           # [K, N_p, 1, Cl, 1]
            probs_targ              = probs_targ[:, None, :, None, :]           # [K, 1, N_t, 1, Cl]
            probs_pool_targ_joint   = probs_pool * probs_targ                   # [K, N_p, N_t, Cl, Cl]
            probs_pool_targ_joint = torch.mean(probs_pool_targ_joint, dim=0)    # [N_p, N_t, Cl, Cl]

            probs_pool = torch.mean(probs_pool, dim=0)          # [N_p, 1, Cl, 1]
            probs_targ = torch.mean(probs_targ, dim=0)          # [1, N_t, 1, Cl]

            probs_pool_targ_indep = probs_pool * probs_targ     # [N_p, N_t, Cl, Cl]

            log_term                = torch.log(probs_pool_targ_joint) - torch.log(probs_pool_targ_indep)
            conditional_acq_scores  = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]
            marginal_acq_scores     = torch.mean(conditional_acq_scores, dim=(-1))  # [N_p]
            return self.order_acq_scores(acq_scores=marginal_acq_scores.numpy(), return_sorted=return_sorted)
        

class GeneralEPIG(AcquisitionFunction):

    def __init__(self, query_n_points, target_input_distribution: TargetInputDistribution, n_posterior_samples: int = 1000, n_target_input_samples: int = 100, seed: int = 0, version: str = 'mine'):
        # Make target-input distribution accesible
        self.target_input_distribution = target_input_distribution

        # Set class-wide sampling parameters
        self.n_posterior_samples    = n_posterior_samples
        self.n_target_input_samples = n_target_input_samples
        self.seed                   = seed
        self.version                = version 

        super().__init__(name='EPIG', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        # Sample x* values from the target input distribution
        Xstar                   = self.target_input_distribution.sample(self.n_target_input_samples, seed=self.seed)
        
        if self.version == 'mine':
            # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior
            probs_pool  = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
            probs_targ  = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)

            # Define constants
            num_classes = probs_pool.shape[0]
            K, M        = self.n_posterior_samples, self.n_target_input_samples

            # Compute the joint term of the expression (summation on the numerator part of the fraction in the log)
            joint_term          = np.array([[(probs_pool[c, :, :] * probs_targ[c_star, :, j][:, None]).sum(axis=0) for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)])
            # Compute the independent term of the expression (summation on the denominator part of the fraction in the log)
            independent_term    = np.array([[probs_pool[c, :, :].sum(axis=0) * probs_targ[c_star, :, j].sum(axis=0) for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)])
            # Compute log-term
            log_term            = np.log(K * joint_term) - np.log(independent_term)
            # Wrap it up and compute the final acquisition scores
            acq_scores          = 1/M * log_term.sum(axis=(1, 0))

            # Sort values
            return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)