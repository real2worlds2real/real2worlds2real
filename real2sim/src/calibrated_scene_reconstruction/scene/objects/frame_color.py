"""No frame says how far over the clip a white is, so a frame's sRGB is read as if the clip were a white of `WHITE`
albedo (white plastic, painted walls: 0.75-0.85): a frame value v is the albedo WHITE * linear(v) -- the brightest the
thing can be; what is darker than white keeps its place relative to white. A clipped channel is only a lower bound
and is not used where a choice depends on it: `match_texture` picks the library texture whose unclipped channels'
ratios are nearest and tints it to them (the clipped channel follows the texture)."""
import numpy as np

WHITE = 0.8   # the albedo a frame's clip stands for
CLIP = 0.96   # sRGB: a channel over it is clipped


def to_linear(srgb):
    srgb = np.asarray(srgb, float)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def albedo(px):
    """(albedo (linear RGB), clipped channels) of an object's pixels (N x 3, 0-255): their median."""
    srgb = np.median(np.asarray(px, float).reshape(-1, 3), 0) / 255.0
    return np.clip(WHITE * to_linear(srgb), 0.0, 1.0), srgb >= CLIP


def match_texture(alb, clipped, names):
    """(name, tint, mean albedo after the tint) of the texture whose mean, tinted by one gain, best matches `alb`
    on the unclipped channels (the log-ratio between them); the tint makes it match them exactly."""
    from calibrated_scene_reconstruction.scene.background.materials import lin, tex_mean

    ok = ~np.asarray(clipped, bool)
    if ok.sum() == 0:
        return None
    best = None
    for n in names:
        m = lin(tex_mean(n))
        lr = np.log(np.maximum(alb[ok], 1e-4)) - np.log(np.maximum(m[ok], 1e-4))
        err = float(np.std(lr)) if ok.sum() > 1 else 0.0
        g = float(np.exp(lr.mean()))
        if best is None or err < best[0]:
            best = (err, n, g, m * g)
    _, n, g, mean = best
    return n, [round(g, 3)] * 3, mean
