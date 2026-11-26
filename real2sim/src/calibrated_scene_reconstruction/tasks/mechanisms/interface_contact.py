"""Measure source-declared mating surfaces in actual native rigid poses."""
from calibrated_scene_reconstruction.assets.interface_geometry import from_poses


def check(ctx,assembly,parts,plans):
    names={p[key] for p in plans for key in ['first','second']}
    poses={name:ctx.pose(assembly+'/'+name) for name in names}
    return from_poses(parts,plans,poses)
