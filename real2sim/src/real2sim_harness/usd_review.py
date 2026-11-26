"""Select the imported asset rather than its simulation world for inspection."""
from __future__ import annotations


def inspection_focus(stage, root_path):
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError('Imported inspection root is missing')
    # Simulator snapshots may include a 100 m support. Remove it before a
    # bounds query; hiding it after fitting the camera leaves tiny assets unseen.
    grounds = [prim for prim in Usd.PrimRange(root)
               if prim.GetName() in {'Ground', 'defaultGroundPlane'}]
    for prim in grounds:
        prim.SetActive(False)
    for name in ('Visuals', 'Asset'):
        candidates = [prim for prim in Usd.PrimRange(root)
                      if prim.GetName() == name and prim.IsActive()]
        if candidates:
            return min(candidates, key=lambda prim: len(prim.GetPath().GetPrefixes()))
    return root
