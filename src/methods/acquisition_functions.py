import math

import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Tuple

import torch
from torch.nn.functional import softmax

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
        if kwargs['model'].__class__.__name__ == 'GaussianProcessClassifier':
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
        # disagreement_term   = (torch.softmax(posterior_samples, dim=0) * (posterior_samples - torch.logsumexp(posterior_samples, dim=0, keepdim=True)))
        # disagreement_term   = disagreement_term.sum(axis=0).mean(axis=0)
        disagreement_term   = (pool_probs_sample * log_probs_sample).sum(axis=0).mean(axis=0)
        
        # Compute final acq-scores
        acq_scores          = entropy_term + disagreement_term
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)

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
        Xstar                       = self.target_input_distribution.sample(M, seed=self.seed)
        self.Xstar                  = Xstar

        # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior 
        posterior_pool_samples      = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
        posterior_target_samples    = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)
        
        if kwargs['model'].__class__.__name__ == 'GaussianProcessClassifier':
            assert posterior_pool_samples.min() != 0,   "Model returns 0 probability for some samples!"
            assert posterior_target_samples.min() != 0, "Model returns 0 probability for some samples!"

            # model returns probabilities - so return log-probs with log
            logprobs_pool               = torch.log(torch.FloatTensor(posterior_pool_samples))[:, None, :, :, None]
            logprobs_target             = torch.log(torch.FloatTensor(posterior_target_samples))[None, :, :, None, :]
        else:
            # model returns logits - so return log-probs with log_softmax
            logprobs_pool               = torch.log_softmax(posterior_pool_samples, dim=0)[:, None, :, :, None]     
            logprobs_target             = torch.log_softmax(posterior_target_samples, dim=0)[None, :, :, None, :]

        # Compute acq_scores per point in the pool as the other approach scales very bad...
        acq_scores = torch.zeros(len(Xpool))
        for idx in tqdm(range(len(Xpool)), desc='Estimating EPIG...'):
            logprobs_pool_              = logprobs_pool[:, :, :, idx, :].unsqueeze(3) 
            logprobs_joint              = logprobs_pool_ + logprobs_target 
            logprobs_joint              = torch.logsumexp(logprobs_joint, dim=2) - math.log(K) # [Cl, Cl, Np, Nt]
            probs_joint                 = torch.exp(logprobs_joint)
        
            logprobs_independent        = (torch.logsumexp(logprobs_pool_, dim=2) - math.log(K)) + (torch.logsumexp(logprobs_target, dim=2) - math.log(K))
            log_term                    = logprobs_joint - logprobs_independent

            acq_scores[idx]             = (probs_joint * log_term).sum(dim=[0, 1]).mean(dim=-1)

        # logprobs_joint              = logprobs_pool + logprobs_target
        # logprobs_joint              = torch.logsumexp(logprobs_joint, dim=2) - math.log(K) # [Cl, Cl, Np, Nt]
        # probs_joint                 = torch.exp(logprobs_joint)

        # logprobs_independent        = (torch.logsumexp(logprobs_pool, dim=2) - math.log(K)) + (torch.logsumexp(logprobs_target, dim=2) - math.log(K))
        # log_term                    = logprobs_joint - logprobs_independent

        # acq_scores                  = (probs_joint * log_term).sum(dim=[0, 1])
        # acq_scores                  = acq_scores.mean(dim=-1)
        assert torch.all((acq_scores + 1e-6 >= 0) & (acq_scores <= math.inf)).item(), "Acquisition scores are not valid!"
        
        # Sort values
        return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)
    
    # def logmeanexp(x, dim: int, keepdim: bool = False):
    #     """
    #     Arguments:
    #         x: Tensor[float]
    #         dim: int
    #         keepdim: bool

    #     Returns:
    #         Tensor[float]
    #     """
    #     return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(x.shape[dim])


    # logprobs_pool               = torch.log_softmax(posterior_pool_samples, dim=0)
    # logprobs_targ             = torch.log_softmax(posterior_target_samples, dim=0)

    # # Estimate the log of the joint predictive distribution.
    # logprobs_pool = logprobs_pool.permute(1, 2, 0)  # [K, N_p, Cl]
    # logprobs_targ = logprobs_targ.permute(1, 2, 0)  # [K, N_t, Cl]
    # logprobs_pool = logprobs_pool[:, :, None, :, None]  # [K, N_p, 1, Cl, 1]
    # logprobs_targ = logprobs_targ[:, None, :, None, :]  # [K, 1, N_t, 1, Cl]
    # logprobs_pool_targ_joint = logprobs_pool + logprobs_targ  # [K, N_p, N_t, Cl, Cl]
    # logprobs_pool_targ_joint = logmeanexp(logprobs_pool_targ_joint, dim=0)  # [N_p, N_t, Cl, Cl]

    # # Estimate the log of the marginal predictive distributions.
    # logprobs_pool = logmeanexp(logprobs_pool, dim=0)  # [N_p, 1, Cl, 1]
    # logprobs_targ = logmeanexp(logprobs_targ, dim=0)  # [1, N_t, 1, Cl]

    # # Estimate the log of the product of the marginal predictive distributions.
    # logprobs_pool_targ_joint_indep = logprobs_pool + logprobs_targ  # [N_p, N_t, Cl, Cl]

    # # Estimate the conditional expected predictive information gain for each pair of examples.
    # # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
    # probs_pool_targ_joint = torch.exp(logprobs_pool_targ_joint)  # [N_p, N_t, Cl, Cl]
    # log_term = logprobs_pool_targ_joint - logprobs_pool_targ_joint_indep  # [N_p, N_t, Cl, Cl]
    # scores = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]
    # return scores  # [N_p, N_t]


# class GeneralEPIG(AcquisitionFunction):

#     def __init__(self, query_n_points, target_input_distribution: TargetInputDistribution, n_posterior_samples: int = 1000, n_target_input_samples: int = 100, seed: int = 0):
#         # Make target-input distribution accesible
#         self.target_input_distribution = target_input_distribution

#         # Set class-wide sampling parameters
#         self.n_posterior_samples    = n_posterior_samples
#         self.n_target_input_samples = n_target_input_samples
#         self.seed                   = seed

#         super().__init__(name='EPIG', query_n_points=query_n_points)

#     def __call__(self, Xpool: np.ndarray, return_sorted: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
#         K, M        = self.n_posterior_samples, self.n_target_input_samples

#         # Sample x* values from the target input distribution
#         Xstar                       = self.target_input_distribution.sample(M, seed=self.seed)
#         self.Xstar                  = Xstar

#         # Extract predictive probabilities for target samples and all points in the pool by exploiting Monte Carlo sampling of the posterior 
#         posterior_pool_samples      = kwargs['model'].sample(np.vstack(Xpool), n_samples=self.n_posterior_samples, seed=self.seed)
#         posterior_target_samples    = kwargs['model'].sample(np.vstack(Xstar), n_samples=self.n_posterior_samples, seed=self.seed)
        
#         if kwargs['model'].__class__.__name__ == 'GaussianProcessClassifier':
#             assert posterior_pool_samples.min() != 0, "Model returns 0 probability for some samples!"
#             assert posterior_target_samples.min() != 0, "Model returns 0 probability for some samples!"

#             # model returns probabilities - so return log-probs with log
#             logprobs_pool               = torch.log(torch.FloatTensor(posterior_pool_samples))[:, None, :, :, None]
#             logprobs_target             = torch.log(torch.FloatTensor(posterior_target_samples))[None, :, :, None, :]
#         else:
#             # model returns logits - so return log-probs with log_softmax
#             logprobs_pool               = torch.log_softmax(posterior_pool_samples, dim=0)[:, None, :, :, None]     
#             logprobs_target             = torch.log_softmax(posterior_target_samples, dim=0)[None, :, :, None, :]

#         # Define distribution and sample targets
#         dist = torch.distributions.Categorical
#         ystar_samples               = dist(probs=torch.exp(logprobs_target)[0, :, :, 0, :].permute(2, 1, 0)).sample()

#         # Compute acq_scores per point in the pool as the other approach scales very bad...
#         acq_scores = torch.zeros(len(Xpool))
#         for idx in tqdm(range(len(Xpool)), desc='Estimating EPIG...'):

#             logprobs_pool_              = logprobs_pool[:, :, :, idx, :].unsqueeze(3) 
#             y_samples                   = torch.vstack([dist(probs=torch.exp(logprobs_pool_)[:, 0, :, 0, 0].T).sample() for _ in range(ystar_samples.shape[0])])

#             logprobs_pool_              = torch.hstack([torch.stack([logprobs_pool_[val, :, j, :, :] for j, val in enumerate(sample_)]) for i, sample_ in enumerate(y_samples)]).squeeze([2, 3])

#             logprobs_target_            = torch.hstack([torch.stack([logprobs_target[:, val, j, :, i] for j, val in enumerate(sample_)]) for i, sample_ in enumerate(ystar_samples)]).squeeze(2)

#             logprobs_joint              = logprobs_pool_ + logprobs_target_
            
#             log_term        = math.log(K) + torch.logsumexp(logprobs_joint, dim=0) - torch.logsumexp(logprobs_pool_, dim=0) - torch.logsumexp(logprobs_target_, dim=0)
#             acq_scores[idx] = log_term.mean(dim=-1)

#         assert torch.all((acq_scores + 1e-6 >= 0) & (acq_scores <= math.inf)).item(), "Acquisition scores are not valid!"
        
#         # Sort values
#         return self.order_acq_scores(acq_scores=acq_scores, return_sorted=return_sorted)
    