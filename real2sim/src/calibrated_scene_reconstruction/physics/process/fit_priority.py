"""Fit-only optimization priorities for plans with mandatory acceptance channels."""
import numpy as np

from .contract import require


def score(metrics,mode='feasibility'):
    """Prefer satisfying every fit channel; held-out evidence is never read.

    Keep the original summed training loss available for backward-compatible
    experiments. Reported fit losses and final validation criteria are unchanged.
    """
    require(mode in {'feasibility','training_loss'},'Unknown process search objective')
    loss=float(metrics['fit_loss'])
    require(np.isfinite(loss) and loss>=0,'Finite nonnegative training loss required')
    if mode=='training_loss':return loss
    ratios=np.array([row['fit_rmse']/row['tolerance'] for row in metrics['channels'].values()])
    require(len(ratios)>0 and np.isfinite(ratios).all() and (ratios>=0).all(),'Finite normalized fit channel errors required')
    normalized=loss/(1+loss);excess=np.maximum(ratios-1,0)
    if not np.any(excess):return normalized
    return float(1+excess.max()+np.mean(excess**2)+.01*normalized)
