"""Procedural PBR texture sets for the surfaces Poly Haven's scans do not cover well: office / lab carpet tiles,
laminate desk tops, speckled vinyl and epoxy floors, clean painted drywall. Written in Poly Haven's layout (diff / nor_gl / rough) under
assets/polyhaven/textures/<name>/, so every user of a texture name (recon.surfaces' candidates, the agent's looks,
style themes, sim.usd.pbr_material) takes them as any scan.

    python -m calibrated_scene_reconstruction.assets.proc_textures [--force]

Each set spans SIZE[name] metres per repeat (the tile_m a look should use; recon.materials reads it).
"""
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter

from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR
from .receptacles import _unit, _write_set

S = 1024  # texels per repeat


def _carpet_tile(rgb, rng, tile=1.0, n_tiles=2):
    """Loop-pile carpet tiles (n_tiles x n_tiles per repeat): pile noise at the loops' scale, darker and lighter yarn
    flecks (the speckle office carpet reads as), each tile's pile laid a quarter turn from its neighbour's (a slight
    shade step), thin seams."""
    px = tile / S
    pile = _unit(gaussian_filter(rng.random((S, S)), 0.0006 / px, mode="wrap"))
    loops = _unit(gaussian_filter(rng.random((S, S)), (0.0004 / px, 0.0014 / px), mode="wrap"))
    yarn = gaussian_filter(rng.random((S, S)), 0.0006 / px, mode="wrap")
    dark, light = yarn < np.quantile(yarn, 0.07), yarn > np.quantile(yarn, 0.95)
    y, x = np.mgrid[0:S, 0:S]
    k = S // n_tiles
    ti, tj = y // k, x // k
    turn = ((ti + tj) % 2).astype(float)
    shade = 1 + 0.035 * (turn - 0.5) + 0.02 * rng.standard_normal((n_tiles, n_tiles))[ti, tj]
    seam = (np.minimum(y % k, k - 1 - y % k) < 1) | (np.minimum(x % k, k - 1 - x % k) < 1)
    tone = shade * (1 + 0.16 * pile + 0.08 * loops) * np.where(dark, 0.55, 1.0) * np.where(light, 1.45, 1.0)
    tone = tone * np.where(seam, 0.8, 1.0)
    albedo = np.asarray(rgb)[None, None] * tone[..., None]
    height = 0.0007 * pile + 0.0004 * loops - 0.0008 * seam
    return albedo, height, np.full((S, S), 0.95)


def _laminate(rgb, rng, tile=1.0):
    """A plain laminate / melamine top: faint low-frequency mottling, a fine print grain, slight sheen."""
    px = tile / S
    mottle = _unit(gaussian_filter(rng.random((S, S)), 0.02 / px, mode="wrap"))
    grain = _unit(gaussian_filter(rng.random((S, S)), 0.0004 / px, mode="wrap"))
    albedo = np.asarray(rgb)[None, None] * (1 + 0.015 * mottle + 0.012 * grain)[..., None]
    return albedo, 0.00002 * grain, 0.38 + 0.03 * mottle


def _vinyl(rgb, rng, tile=1.0):
    """Speckled commercial vinyl / linoleum sheet: chips of lighter and darker shades in a mid tone."""
    px = tile / S
    base = _unit(gaussian_filter(rng.random((S, S)), 0.004 / px, mode="wrap"))
    chips = gaussian_filter(rng.random((S, S)), 0.0012 / px, mode="wrap")
    light, dark = chips > np.quantile(chips, 0.93), chips < np.quantile(chips, 0.05)
    tone = (1 + 0.04 * base) * np.where(light, 1.25, 1.0) * np.where(dark, 0.7, 1.0)
    albedo = np.asarray(rgb)[None, None] * tone[..., None]
    return albedo, 0.00001 * base, 0.45 + 0.05 * base


def _epoxy(rgb, rng, tile=2.0):
    """A sealed epoxy / painted concrete floor: broad trowel clouds, fine flakes, a low gloss."""
    px = tile / S
    clouds = _unit(gaussian_filter(rng.random((S, S)), 0.05 / px, mode="wrap"))
    flakes = gaussian_filter(rng.random((S, S)), 0.001 / px, mode="wrap") > 0.53
    tone = (1 + 0.05 * clouds) * np.where(flakes, 0.92, 1.0)
    albedo = np.asarray(rgb)[None, None] * tone[..., None]
    return albedo, 0.00002 * clouds, 0.3 + 0.05 * clouds


def _drywall(rgb, rng, tile=1.0):
    """Painted drywall: a roller's fine orange-peel stipple, faint low-frequency unevenness, an eggshell sheen; clean
    (a scanned plaster brings its stains and moss along)."""
    px = tile / S
    peel = _unit(gaussian_filter(rng.random((S, S)), 0.0009 / px, mode="wrap"))
    uneven = _unit(gaussian_filter(rng.random((S, S)), 0.08 / px, mode="wrap"))
    albedo = np.asarray(rgb)[None, None] * (1 + 0.012 * peel + 0.01 * uneven)[..., None]
    return albedo, 0.00015 * peel, 0.7 + 0.04 * peel


# name: (maker, linear rgb, seed, metres per repeat)
SETS = {
    "proc_carpet_tile_grey": (_carpet_tile, (0.16, 0.16, 0.165), 1, 1.0),
    "proc_carpet_tile_bluegrey": (_carpet_tile, (0.12, 0.14, 0.18), 2, 1.0),
    "proc_carpet_tile_charcoal": (_carpet_tile, (0.06, 0.06, 0.065), 3, 1.0),
    "proc_carpet_tile_beige": (_carpet_tile, (0.28, 0.25, 0.20), 4, 1.0),
    "proc_laminate_white": (_laminate, (0.80, 0.80, 0.78), 5, 1.0),
    "proc_laminate_grey": (_laminate, (0.42, 0.43, 0.44), 6, 1.0),
    "proc_laminate_black": (_laminate, (0.035, 0.035, 0.037), 7, 1.0),
    "proc_vinyl_grey": (_vinyl, (0.30, 0.30, 0.30), 8, 1.0),
    "proc_vinyl_blue": (_vinyl, (0.16, 0.20, 0.27), 9, 1.0),
    "proc_epoxy_grey": (_epoxy, (0.30, 0.31, 0.32), 10, 2.0),
    "proc_painted_drywall": (_drywall, (0.78, 0.78, 0.77), 11, 1.0),
}
SIZE = {name: v[3] for name, v in SETS.items()}


def write(name, force=False):
    maker, rgb, seed, tile = SETS[name]
    d = POLYHAVEN_DIR / "textures" / name
    if (d / "rough.jpg").exists() and not force:
        return d
    albedo, height, rough = maker(rgb, np.random.default_rng(seed), tile)
    _write_set(d, albedo ** (1 / 2.2), height, rough, tile / S)  # diff.jpg is sRGB like a scan's
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for name in SETS:
        print("texture", name, write(name, a.force))


if __name__ == "__main__":
    main()
