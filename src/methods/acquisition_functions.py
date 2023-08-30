import numpy as np
from typing import Optional, Union, Tuple

class AcquisitionFunction:
    def __init__(self, name: str, query_n_points: Optional[int] = None):
        self.name           = name
        self.query_n_points = query_n_points

    def __call__(self, value: Union[int, float]):
        return None

class VariationRatios(AcquisitionFunction):

    def __init__(self, query_n_points):
        super().__init__(name='Variation Ratios', query_n_points=query_n_points)

    def __call__(self, pool_probs: np.ndarray, return_sorted: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # Compute Variation Ratios and sort
        acq_scores      = 1 - pool_probs.max(axis=1)
        if return_sorted:
            pairs       = list(zip(*sorted(zip(acq_scores, np.arange(len(pool_probs))), key=lambda x: x[0], reverse=True)))
        else:
            pairs       = [acq_scores, np.arange(len(pool_probs))]

        # return N maximum Variation Ratios
        max_acq_scores  = np.array(pairs[0][:self.query_n_points])
        query_idxs      = np.array(pairs[1][:self.query_n_points])
        return max_acq_scores, query_idxs 

class BALD(AcquisitionFunction):

    def __init__(self, query_n_points):
        super().__init__(name='BALD', query_n_points=query_n_points)

    def __call__(self, value: Union[int, float]):
        return value

class EPIG(AcquisitionFunction):

    def __init__(self, query_n_points):
        super().__init__(name='EPIG', query_n_points=query_n_points)

    def __call__(self, value: Union[int, float]):
        return value
