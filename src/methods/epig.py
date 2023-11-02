import torch
from tqdm import tqdm
import math

def epig_from_logprobs(Xpool, K, posterior_pool_samples, posterior_target_samples, kwargs):
    if kwargs['model'].__class__.__name__ in ['BinaryGenerativeGaussian', 'BayesianLogisticRegression', 'GaussianProcessClassifier']:
        # model returns probabilities - so return log-probs with log
        logprobs_pool               = torch.log(torch.FloatTensor(posterior_pool_samples))[:, None, :, :, None]
        logprobs_target             = torch.log(torch.FloatTensor(posterior_target_samples))[None, :, :, None, :]
    else:
        # model returns logits - so return log-probs with log_softmax
        logprobs_pool               = torch.log_softmax(posterior_pool_samples, dim=0)[:, None, :, :, None]     
        logprobs_target             = torch.log_softmax(posterior_target_samples, dim=0)[None, :, :, None, :]

    # Compute acq_scores per point in the pool as the other approach scales very bad for memory...
    acq_scores = torch.zeros(len(Xpool))
    
    for idx in tqdm(range(len(Xpool)), desc='Estimating EPIG...'):
        logprobs_pool_              = logprobs_pool[:, :, :, idx, :].unsqueeze(3) 
        logprobs_joint              = logprobs_pool_ + logprobs_target 
        logprobs_joint              = torch.logsumexp(logprobs_joint, dim=2) - math.log(K) # [Cl, Cl, Np, Nt]
        probs_joint                 = torch.exp(logprobs_joint)
    
        logprobs_independent        = (torch.logsumexp(logprobs_pool_, dim=2) - math.log(K)) + (torch.logsumexp(logprobs_target, dim=2) - math.log(K))
        log_term                    = logprobs_joint - logprobs_independent

        acq_scores[idx]             = (probs_joint * log_term).sum(dim=[0, 1]).mean(dim=-1)

    # torch.save(acq_scores, r'C:\Users\alber\Desktop\DTU\3_HCAI\ActiveBayesianDeepLearning\abdl\notebooks\acq_scores.pt')
    return acq_scores

def epig_from_probs(Xpool, K, posterior_pool_samples, posterior_target_samples, kwargs):
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

    return acq_scores