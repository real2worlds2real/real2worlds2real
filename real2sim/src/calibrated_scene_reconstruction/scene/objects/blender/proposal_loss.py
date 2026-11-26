"""Source proposal gradients with the original local-component gate value.

The hard forward value uses the same morphology and per-component cutoff as
CPU acceptance. A soft opening supplies gradients only in retained regions.
This is a search aid: exact CPU source checks still select/approve geometry.
"""
import numpy as np


def local_components(predicted, target, domain, *, hard_prediction=None, share=.003):
    """Return exact ``pieces`` share with a local differentiable surrogate.

    ``hard_prediction`` can carry the actual raster's sample coverage. Using
    anti-aliased coverage alone to pick components would change the hard
    silhouette at an edge, so gradient renderers should pass both explicitly.
    Tiny connected components retain their old status in the pixel loss; they
    cannot silently become extra failures in the separate local-shape loss.
    """
    import torch
    import torch.nn.functional as functional
    from calibrated_scene_reconstruction.scene.objects.object_refine import pieces

    if (predicted.ndim != 2 or predicted.shape != target.shape or predicted.shape != domain.shape
            or not np.isfinite(share) or not 0 <= share <= 1):
        raise ValueError('Matching image/domain shapes and a bounded component share required')
    hard = predicted.detach() >= .5 if hard_prediction is None else hard_prediction.detach().bool()
    if hard.shape != predicted.shape:
        raise ValueError('Hard sample coverage must match the differentiable raster')
    known = domain.detach().bool()
    truth = target.detach().bool().cpu().numpy()
    regions = pieces(truth, (hard & known).cpu().numpy(), share)
    exact = sum(row[0] for row in regions)
    surrogate = predicted.sum() * 0
    for sign, difference in (('adds', predicted-target), ('lacks', target-predicted)):
        kept = np.zeros(truth.shape, bool)
        for _, mask, which in regions:
            if which == sign:
                kept |= mask
        if not kept.any():
            continue
        positive = difference.clamp(min=0) * domain
        eroded = -functional.max_pool2d(-positive[None, None], 3, stride=1, padding=1)
        opened = functional.max_pool2d(eroded, 3, stride=1, padding=1)[0, 0]
        selected = torch.as_tensor(kept, device=predicted.device, dtype=predicted.dtype)
        surrogate = surrogate + (opened * selected).sum() / target.sum().clamp(min=1)
    return surrogate + (predicted.new_tensor(exact) - surrogate.detach())
