"""Read actual legacy FEM collision boundaries for solid-contact checks."""
import numpy as np

from calibrated_scene_reconstruction.assets.tetra_boundary import surface,determinants
from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface


class LegacySurface:
    def __init__(self, view, prim, *, device='cuda:0'):
        from pxr import PhysxSchema,UsdGeom
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        self.view=view
        self.indices=as_numpy(view.get_simulation_mesh_indices())[0].astype(int)
        api=PhysxSchema.PhysxDeformableBodyAPI(prim)
        self.simulation_rest_points=np.asarray(api.GetSimulationRestPointsAttr().Get(),float).copy()
        self.collision_rest_points=np.asarray(api.GetCollisionRestPointsAttr().Get(),float).copy()
        self.local_to_world_row_matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim),float).copy()
        self.rest_determinants=determinants(self.simulation_rest_points,self.indices)
        if np.any(abs(self.rest_determinants)<1e-20):raise ValueError('Native soft-solid rest cells are degenerate')
        points=as_numpy(view.get_collision_mesh_nodal_positions())[0]
        tets=as_numpy(view.get_collision_mesh_indices())[0].astype(int)
        self.collision_indices=tets.copy()
        self.faces=surface(points,tets)
        self.distance=SignedSurface(points,self.faces,device=device)
        self.points=points.copy()
        self.refresh()

    def refresh(self):
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        sim=as_numpy(self.view.get_simulation_mesh_nodal_positions())[0]
        self.simulation_points=sim.copy()
        jacobian=determinants(sim,self.indices)/self.rest_determinants
        if not np.isfinite(jacobian).all() or (jacobian<=0).any():
            raise ValueError('Native soft solid has inverted or nonfinite material cells')
        self.points=as_numpy(self.view.get_collision_mesh_nodal_positions())[0].copy()
        self.distance.update(self.points)
        return {'minimum_volume_jacobian':float(jacobian.min()),
                'maximum_volume_jacobian':float(jacobian.max()),
                'collision_nodes':len(self.points),'boundary_triangles':len(self.faces),
                'native_collision_geometry':True,'fixed_rest_mesh_used':False}

    def reference(self):
        """Read material reference separately from the current deformed state.

        Coordinates are serialized explicitly; this performs no rest-shape or
        native-position writes and does not qualify export/reload by itself.
        """
        return {'simulation_rest_points_local':self.simulation_rest_points.copy(),
                'collision_rest_points_local':self.collision_rest_points.copy(),
                'simulation_tetrahedra':self.indices.copy(),
                'collision_tetrahedra':self.collision_indices.copy(),
                'local_to_world_row_matrix':self.local_to_world_row_matrix.copy()}


class VolumeSurface(LegacySurface):
    def __init__(self,view,prim,*,device='cuda:0'):
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        self.view=view;self.indices=view.indices.copy()
        self.simulation_rest_points=view.rest_local.copy()
        self.collision_rest_points=self.simulation_rest_points.copy()
        self.local_to_world_row_matrix=view.initial_transform.copy()
        self.rest_determinants=determinants(self.simulation_rest_points,self.indices)
        if np.any(abs(self.rest_determinants)<1e-20):raise ValueError('Native volume reference cells are degenerate')
        self.collision_indices=self.indices.copy()
        points=as_numpy(view.get_collision_mesh_nodal_positions())[0]
        self.faces=surface(points,self.indices);self.distance=SignedSurface(points,self.faces,device=device)
        self.points=points.copy();self.refresh()


def from_view(view,prim,*,device='cuda:0'):
    cls=VolumeSurface if prim.HasAPI('OmniPhysicsVolumeDeformableSimAPI') else LegacySurface
    return cls(view,prim,device=device)
