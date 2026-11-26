"""Differentiable proposal objectives; acceptance uses exact CPU evidence."""
import torch
import torch.nn.functional as nnf


def balanced_part_loss(relative_errors):
    """Retain every view while also addressing the unweighted median gate.

    Native-pixel pooling can improve large views while leaving the median
    source-view error worse. Equal-view mean plus median retains gradients
    for every view. This proxy is not a connected-component measurement.
    """
    if (relative_errors.ndim!=1 or not len(relative_errors)
            or not torch.isfinite(relative_errors).all() or (relative_errors<0).any()):
        raise ValueError('Finite nonnegative per-view relative errors required')
    return relative_errors.mean()+torch.quantile(relative_errors,.5)


def local_part_residual(prediction,target,domain):
    """Soft two-sided 3px opening proxy for persistent outline errors.

    Detached source labels and visibility domains stay fixed. Thin antialias
    boundaries must not dominate broader missing/extra regions. Exact CPU
    connected components, their size filter and acceptance are unchanged.
    Every view still contributes through the separate global residual.
    """
    if (prediction.ndim!=3 or target.shape!=prediction.shape or domain.shape!=prediction.shape
            or not all(torch.isfinite(x).all() for x in [prediction,target,domain])):
        raise ValueError('Matching finite per-view masks and source domains required')
    residuals=[]
    for difference in [torch.relu(target-prediction),torch.relu(prediction-target)]:
        difference=difference*domain
        eroded=-nnf.max_pool2d(-difference[:,None],3,stride=1,padding=1)
        opened=nnf.max_pool2d(eroded,3,stride=1,padding=1)[:,0]*domain
        residuals.append(opened.sum((1,2)))
    return sum(residuals)/(target*domain).sum((1,2)).clamp_min(1.)
