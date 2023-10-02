import numpy as np
import torch
from torch.nn.functional import softmax

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
        posterior_samples   = kwargs['model'].sample(Xpool, n_samples=self.n_posterior_samples, seed=self.seed)
        pool_probs          = torch.softmax(posterior_samples, dim=0).mean(axis=1).T  

        ### BALD score estimation as of Gal, et al.: https://arxiv.org/pdf/1703.02910.pdf 
        # Compute entropy term
        log_softmax         = (posterior_samples - torch.logsumexp(posterior_samples, dim=0, keepdim=True)).mean(axis=1).T
        entropy_term        = - sum([pool_probs[:, cat] * log_softmax[:, cat] for cat in range(pool_probs.shape[1])])

        # Sample the posterior and compute disagreement term
        disagreement_term   = (torch.softmax(posterior_samples, dim=0) * (posterior_samples - torch.logsumexp(posterior_samples, dim=0, keepdim=True)))
        disagreement_term   = disagreement_term.sum(axis=0).mean(axis=0)

        # Compute final acq-scores
        acq_scores          = entropy_term + disagreement_term
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

import math
class EPIG(AcquisitionFunction):

    def __init__(self, query_n_points, target_input_distribution: TargetInputDistribution, n_posterior_samples: int = 1000, n_target_input_samples: int = 100, seed: int = 0):
        # Make target-input distribution accesible
        self.target_input_distribution = target_input_distribution

        # Set class-wide sampling parameters
        self.n_posterior_samples    = n_posterior_samples
        self.n_target_input_samples = n_target_input_samples
        self.seed                   = seed

        super().__init__(name='EPIG', query_n_points=query_n_points)

    def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        K, M        = self.n_posterior_samples, self.n_target_input_samples

        # Sample x* values from the target input distribution
        Xstar                       = self.target_input_distribution.sample(self.n_target_input_samples, seed=self.seed)
        self.Xstar                  = Xstar

        # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior
        posterior_pool_samples      = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
        posterior_target_samples    = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)

        logprobs_pool               = torch.log_softmax(posterior_pool_samples, dim=0)[:, None, :, :, None]
        logprobs_target             = torch.log_softmax(posterior_target_samples, dim=0)[None, :, :, None, :]

        logprobs_joint              = logprobs_pool + logprobs_target
        logprobs_joint              = torch.logsumexp(logprobs_joint, dim=2) - math.log(K)
        probs_joint                 = torch.exp(logprobs_joint)

        logprobs_independent        = (torch.logsumexp(logprobs_pool, dim=2) - math.log(K)) + (torch.logsumexp(logprobs_target, dim=2) - math.log(K))
        log_term                    = logprobs_joint - logprobs_independent

        acq_scores                  = (probs_joint * log_term).sum(dim=[0, 1])
        acq_scores                  = acq_scores.mean(dim=-1)
        assert torch.all((acq_scores + 1e-6 >= 0) & (acq_scores - 1e-6 <= math.inf)).item(), "Acquisition scores are not valid!"
        
        # Sort values
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)
    

# class EPIG(AcquisitionFunction):

#     def __init__(self, query_n_points, target_input_distribution: TargetInputDistribution, n_posterior_samples: int = 1000, n_target_input_samples: int = 100, seed: int = 0, version: str = 'mine'):
#         # Make target-input distribution accesible
#         self.target_input_distribution = target_input_distribution

#         # Set class-wide sampling parameters
#         self.n_posterior_samples    = n_posterior_samples
#         self.n_target_input_samples = n_target_input_samples
#         self.seed                   = seed
#         self.version                = version 

#         super().__init__(name='EPIG', query_n_points=query_n_points)

#     def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

#         # Sample x* values from the target input distribution
#         Xstar                   = self.target_input_distribution.sample(self.n_target_input_samples, seed=self.seed)
        
#         if self.version == 'mine':
#             # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior
#             _, probs_pool  = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
#             _, probs_targ  = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)

#             # Define constants
#             num_classes = probs_pool.shape[0]
#             K, M        = self.n_posterior_samples, self.n_target_input_samples

#             # Compute the joint term of the expression (summation on the numerator part of the fraction in the log)
#             joint_term          = np.array([[(probs_pool[c, :, :] * probs_targ[c_star, :, j][:, None]).sum(axis=0) for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)])
#             # Compute the independent term of the expression (summation on the denominator part of the fraction in the log)
#             independent_term    = np.array([[probs_pool[c, :, :].sum(axis=0) * probs_targ[c_star, :, j].sum(axis=0) for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)])
#             # Compute log-term
#             log_term            = np.log(K * joint_term) - np.log(independent_term)
#             # Wrap it up and compute the final acquisition scores
#             acq_scores          = 1/M * (joint_term/K * log_term).sum(axis=(1, 0)) #.sum(axis=0)

#             # Sort values
#             return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

#         else: # outdated - found from their repository - it is quite slow, compared to my implementation and yielded the same result
#             ### TAKEN FROM THE AUTHORS OF THE EPIG PAPER: https://github.com/fbickfordsmith/epig/blob/main/src/uncertainty/bald.py 
#             probs_pool  = torch.tensor(kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed))
#             probs_targ  = torch.tensor(kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed))

#             probs_pool = probs_pool.permute(1, 2, 0)  # [K, N_p, Cl]
#             probs_targ = probs_targ.permute(1, 2, 0)  # [K, N_t, Cl]

#             probs_pool              = probs_pool[:, :, None, :, None]           # [K, N_p, 1, Cl, 1]
#             probs_targ              = probs_targ[:, None, :, None, :]           # [K, 1, N_t, 1, Cl]
#             probs_pool_targ_joint   = probs_pool * probs_targ                   # [K, N_p, N_t, Cl, Cl]
#             probs_pool_targ_joint = torch.mean(probs_pool_targ_joint, dim=0)    # [N_p, N_t, Cl, Cl]

#             probs_pool = torch.mean(probs_pool, dim=0)          # [N_p, 1, Cl, 1]
#             probs_targ = torch.mean(probs_targ, dim=0)          # [1, N_t, 1, Cl]

#             probs_pool_targ_indep = probs_pool * probs_targ     # [N_p, N_t, Cl, Cl]

#             log_term                = torch.log(probs_pool_targ_joint) - torch.log(probs_pool_targ_indep)
#             conditional_acq_scores  = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]
#             marginal_acq_scores     = torch.mean(conditional_acq_scores, dim=(-1))  # [N_p]
#             return self.order_acq_scores(acq_scores=marginal_acq_scores.numpy(), return_sorted=return_sorted)
        
# import math
# class TheirEPIG(AcquisitionFunction):

#     def __init__(self, query_n_points, target_input_distribution: TargetInputDistribution, n_posterior_samples: int = 1000, n_target_input_samples: int = 100, seed: int = 0):
#         # Make target-input distribution accesible
#         self.target_input_distribution = target_input_distribution

#         # Set class-wide sampling parameters
#         self.n_posterior_samples    = n_posterior_samples
#         self.n_target_input_samples = n_target_input_samples
#         self.seed                   = seed

#         super().__init__(name='GeneralEPIG', query_n_points=query_n_points)

#     def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

#         # Sample x* values from the target input distribution
#         Xstar                   = self.target_input_distribution.sample(self.n_target_input_samples, seed=self.seed)
        
#         # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior
#         posterior_pool_samples      = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
#         posterior_target_samples    = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)

#         def logmeanexp(x, dim: int, keepdim: bool = False):
#             """
#             Arguments:
#                 x: Tensor[float]
#                 dim: int
#                 keepdim: bool

#             Returns:
#                 Tensor[float]
#             """
#             return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(x.shape[dim])

#         logprobs_pool = posterior_pool_samples.permute(1, 0, 2)  # [K, N_p, Cl]
#         logprobs_targ = posterior_target_samples.permute(1, 0, 2)  # [K, N_t, Cl]
#         logprobs_pool = logprobs_pool[:, :, None, :, None]  # [K, N_p, 1, Cl, 1]
#         logprobs_targ = logprobs_targ[:, None, :, None, :]  # [K, 1, N_t, 1, Cl]
#         logprobs_pool_targ_joint = logprobs_pool + logprobs_targ  # [K, N_p, N_t, Cl, Cl]
#         logprobs_pool_targ_joint = logmeanexp(logprobs_pool_targ_joint, dim=0)  # [N_p, N_t, Cl, Cl]

#         # Estimate the log of the marginal predictive distributions.
#         logprobs_pool = logmeanexp(logprobs_pool, dim=0)  # [N_p, 1, Cl, 1]
#         logprobs_targ = logmeanexp(logprobs_targ, dim=0)  # [1, N_t, 1, Cl]

#         # Estimate the log of the product of the marginal predictive distributions.
#         logprobs_pool_targ_joint_indep = logprobs_pool + logprobs_targ  # [N_p, N_t, Cl, Cl]

#         # Estimate the conditional expected predictive information gain for each pair of examples.
#         # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
#         probs_pool_targ_joint   = torch.exp(logprobs_pool_targ_joint)  # [N_p, N_t, Cl, Cl]
#         log_term                = logprobs_pool_targ_joint - logprobs_pool_targ_joint_indep  # [N_p, N_t, Cl, Cl]
#         acq_scores              = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]

#         # Sort values
#         return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)


# class MyEPIG(AcquisitionFunction):

#     def __init__(self, query_n_points, target_input_distribution: TargetInputDistribution, n_posterior_samples: int = 1000, n_target_input_samples: int = 100, seed: int = 0):
#         # Make target-input distribution accesible
#         self.target_input_distribution = target_input_distribution

#         # Set class-wide sampling parameters
#         self.n_posterior_samples    = n_posterior_samples
#         self.n_target_input_samples = n_target_input_samples
#         self.seed                   = seed

#         super().__init__(name='GeneralEPIG', query_n_points=query_n_points)

#     def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

#         # Sample x* values from the target input distribution
#         Xstar                   = self.target_input_distribution.sample(self.n_target_input_samples, seed=self.seed)
        
#         # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior
#         posterior_pool_samples      = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
#         posterior_target_samples    = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)

#         pool_probs                  = torch.softmax(posterior_pool_samples, dim=0) #.mean(axis=1).T  
#         targ_probs                  = torch.softmax(posterior_target_samples, dim=0)# .mean(axis=1).T  

#         # Define constants
#         num_classes = pool_probs.shape[0]
#         K, M        = self.n_posterior_samples, self.n_target_input_samples

#         # Compute the joint term of the expression (summation on the numerator part of the fraction in the log)
#         joint_term          = torch.tensor(np.array([[((pool_probs[c, :, :] * targ_probs[c_star, :, j][:, None]).sum(axis=0)).cpu().numpy() for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)]))
#         # Compute the independent term of the expression (summation on the denominator part of the fraction in the log)
#         independent_term    = torch.tensor(np.array([[(pool_probs[c, :, :].sum(axis=0) * targ_probs[c_star, :, j].sum(axis=0)).cpu().numpy() for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)]))
#         # Compute log-term
#         log_term            = np.log(K * joint_term) - np.log(independent_term)
#         # Wrap it up and compute the final acquisition scores
#         acq_scores          = 1/M * log_term.sum(axis=(1, 0))

#         # Sort values
#         return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

# class GeneralEPIG(AcquisitionFunction):

#     def __init__(self, query_n_points, target_input_distribution: TargetInputDistribution, n_posterior_samples: int = 1000, n_target_input_samples: int = 100, seed: int = 0):
#         # Make target-input distribution accesible
#         self.target_input_distribution = target_input_distribution

#         # Set class-wide sampling parameters
#         self.n_posterior_samples    = n_posterior_samples
#         self.n_target_input_samples = n_target_input_samples
#         self.seed                   = seed

#         super().__init__(name='GeneralEPIG', query_n_points=query_n_points)

#     def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

#         # Sample x* values from the target input distribution
#         Xstar                   = self.target_input_distribution.sample(self.n_target_input_samples, seed=self.seed)
        
#         # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior
#         # _, probs_pool  = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
#         # _, probs_targ  = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)
#         posterior_pool_samples      = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
#         targ_probs                  = torch.softmax(posterior_pool_samples, dim=0).mean(axis=1).T  
#         posterior_target_samples    = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)
#         pool_probs                  = torch.softmax(posterior_target_samples, dim=0).mean(axis=1).T  

#         # Define constants
#         num_classes = pool_probs.shape[0]
#         K, M        = self.n_posterior_samples, self.n_target_input_samples

#         # Compute the joint term of the expression (summation on the numerator part of the fraction in the log)
#         joint_term          = np.array([[(pool_probs[c, :, :] * targ_probs[c_star, :, j][:, None]).sum(axis=0) for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)])
#         # Compute the independent term of the expression (summation on the denominator part of the fraction in the log)
#         independent_term    = np.array([[pool_probs[c, :, :].sum(axis=0) * targ_probs[c_star, :, j].sum(axis=0) for j in range(M)] for c in range(num_classes) for c_star in range(num_classes)])
#         # Compute log-term
#         log_term            = np.log(K * joint_term) - np.log(independent_term)
#         # Wrap it up and compute the final acquisition scores
#         acq_scores          = 1/M * log_term.sum(axis=(1, 0))

#         # Sort values
#         return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)