"""Read native conforming volume FEM in its actual cooked material order.

This adapter exposes observations only. It does not provide a positional or
nodal-force task controller and never modifies material references at runtime.
"""
import numpy as np


class VolumeView:
    backend='volume_fem'

    def __init__(self,stage,path,part):
        import omni.physics.tensors as tensors
        from pxr import UsdGeom
        from scipy.spatial import cKDTree
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        self.owner=tensors.create_simulation_view('torch');self.owner.set_subspace_roots('/')
        self.native=self.owner.create_volume_deformable_body_view(path)
        if self.native is None or self.native.count!=1:raise ValueError('One actual native volume FEM body required')
        self.initial_transform=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)),float)
        ref=np.asarray(part['volume_initial']['simulation_rest_points'],float)
        expected=ref@self.initial_transform[:3,:3]+self.initial_transform[3,:3]
        native=as_numpy(self.native.get_rest_nodal_positions())[0]
        distance,mapping=cKDTree(expected).query(native)
        if (native.shape!=expected.shape or len(set(mapping))!=len(expected) or distance.max()>1e-6):
            raise ValueError('Native volume reference lost exact material correspondence')
        cells=as_numpy(self.native.get_simulation_element_indices())[0].astype(int)
        source=np.asarray(part['volume_initial']['simulation_indices'],int)
        if cells.shape!=source.shape or set(map(tuple,np.sort(mapping[cells],axis=1)))!=set(map(tuple,np.sort(source,axis=1))):
            raise ValueError('Native volume reference tetrahedra changed')
        self.indices=cells
        self.rest_local=(native-self.initial_transform[3,:3])@np.linalg.inv(self.initial_transform[:3,:3])

    def get_simulation_mesh_nodal_positions(self):return self.native.get_simulation_nodal_positions()
    def get_collision_mesh_nodal_positions(self):return self.native.get_simulation_nodal_positions()
    def get_simulation_mesh_nodal_velocities(self):return self.native.get_simulation_nodal_velocities()
    def get_simulation_mesh_indices(self):return self.native.get_simulation_element_indices()
    def get_collision_mesh_indices(self):return self.native.get_simulation_element_indices()
