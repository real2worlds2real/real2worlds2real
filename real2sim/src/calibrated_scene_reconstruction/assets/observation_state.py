"""Cached source-solid contact screen for unforced rigid observations."""
import numpy as np
from calibrated_scene_reconstruction.assets.rigid_initial_state import surface_samples
from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface


class RigidState:
    def __init__(self,parts,*,spacing_m,tolerance_m,device):
        if not parts or not np.isfinite(tolerance_m) or tolerance_m<0:
            raise ValueError('Rigid source parts and a nonnegative contact limit required')
        self.parts=parts;self.spacing=spacing_m;self.tolerance=tolerance_m
        self.samples=[surface_samples(p['V'],p['F'],spacing_m) for p in parts] if len(parts)>1 else []
        self.surfaces=[SignedSurface(p['V'],p['F'],device=device) for p in parts] if len(parts)>1 else []

    def check(self,poses):
        if len(poses)!=len(self.parts):raise ValueError('Every rigid source part needs its actual native pose')
        for p,R in poses:
            p,R=np.asarray(p),np.asarray(R)
            if (p.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(R).all()
                    or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1.,atol=1e-5)):
                raise ValueError('Finite proper native rigid poses required')
        pairs=[]
        for i in range(len(poses)):
            for j in range(i):
                depths=[]
                for source,target in [(i,j),(j,i)]:
                    p,R=poses[source];q,S=poses[target]
                    points=(self.samples[source]@R.T+p-q)@S
                    depths.append(self.surfaces[target].max_penetration(points))
                maximum=max(depths)
                pairs.append({'links':[self.parts[i]['name'],self.parts[j]['name']],
                              'maximum_sampled_penetration_m':maximum,'pass':maximum<=self.tolerance})
        return {'pass':all(row['pass'] for row in pairs),'pairs':pairs,'rigid_poses_valid':True,
                'spacing_m':self.spacing,'tolerance_m':self.tolerance,
                'scope':'Finite within-object rigid source-solid screen; assembled task/support contacts require the full task assay.'}
