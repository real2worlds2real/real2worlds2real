"""Intrinsic sheet strain and area measurements, before and during simulation."""
import numpy as np

from .cloth_rest import validate, triangle_edges, material_inverse, principal_strains


class MaterialMetric:
    """Keep material coordinates immutable while measuring authored/native states."""
    def __init__(self, points, faces, rest, *, max_edge_strain, max_area_change, unit_length_m=1.):
        result = validate(rest, points, faces)
        if (not np.isfinite([max_edge_strain, max_area_change]).all()
                or min(max_edge_strain, max_area_change) <= 0):
            raise ValueError('Declare positive finite material strain/area validity limits')
        if isinstance(unit_length_m,bool) or not np.isfinite(unit_length_m) or not 0<unit_length_m<=1:
            raise ValueError('Explicit positive native coordinate unit in metres required')
        self.unit_length_m=float(unit_length_m)
        base = np.asarray(rest['points'], float)
        self.material_points = base.copy()
        self.shape = base.shape
        self.faces = np.asarray(faces).copy()
        from .cloth_topology import measure as surface_topology
        self.topology=surface_topology(len(base),self.faces)
        self.pairs = triangle_edges(faces)
        self.lengths = np.linalg.norm(base[self.pairs[:, 0]]-base[self.pairs[:, 1]], axis=1)
        self.rest_inverse = material_inverse(base, self.faces)
        self.rest_area = result['rest_area_m2']
        self.limits = {'max_edge_strain': float(max_edge_strain), 'max_area_change': float(max_area_change)}

    def bond_diagnostic(self, points, indices, targets):
        """Locate attachment error without excluding any material from checks.

        Small absolute bond errors can be large compared with thin root cells.
        This diagnoses the representation/solver, never identifies a material
        modulus or grants permission to discard strain inside a real bond.
        """
        p=np.asarray(points,float);ids=np.asarray(indices);q=np.asarray(targets,float)
        if (p.shape!=self.shape or ids.ndim!=1 or ids.dtype.kind not in 'iu'
                or len(ids)<3 or len(np.unique(ids))!=len(ids)
                or ids.min()<0 or ids.max()>=len(p) or q.shape!=(len(ids),3)
                or not np.isfinite(p).all() or not np.isfinite(q).all()):
            raise ValueError('Actual sheet nodes and distinct finite physical bond targets required')
        membership=np.isin(self.faces,ids).sum(1)
        strain=abs(principal_strains(p,self.faces,self.rest_inverse)).max(1)
        triangles=self.material_points[self.faces]
        longest=np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2).max(1)
        altitude=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/longest
        error=float(np.linalg.norm(p[ids]-q,axis=1).max())
        root_altitude=float(altitude[membership>0].min())
        regions={}
        for name,selected in [('fully_bonded',membership==3),('bond_boundary',(membership>0)&(membership<3)),('free',membership==0)]:
            regions[name]={'triangles':int(selected.sum()),
                'maximum_principal_material_strain':float(strain[selected].max()) if selected.any() else None}
        return {'regions':regions,'maximum_bond_error_m':error*self.unit_length_m,
                'minimum_bond_incident_material_altitude_m':root_altitude*self.unit_length_m,
                'bond_error_over_minimum_material_altitude':error/root_altitude,
                'acceptance_changed':False,'material_excluded_from_acceptance':False,
                'scope':'Diagnostic localization only. All faces retain the original material check; the ratio is a resolution indicator, not a constitutive parameter or a rigorous strain-error bound.'}

    def check(self, points):
        points = np.asarray(points, float)
        if points.shape != self.shape or not np.isfinite(points).all():
            raise ValueError('Nonfinite or mismatched native cloth state')
        current = np.linalg.norm(points[self.pairs[:, 0]]-points[self.pairs[:, 1]], axis=1)
        strain = current/self.lengths-1
        faces = self.faces
        areas = np.linalg.norm(np.cross(points[faces[:, 1]]-points[faces[:, 0]],
                                        points[faces[:, 2]]-points[faces[:, 0]]), axis=1)/2
        area = float(areas.sum())
        change = area/self.rest_area-1
        maximum = float(np.max(abs(strain)))
        principal = principal_strains(points, faces, self.rest_inverse)
        magnitude = abs(principal).max(axis=1)
        membrane_max = float(magnitude.max())
        count=min(8,len(faces));selected=np.argpartition(magnitude,-count)[-count:]
        selected=selected[np.argsort(magnitude[selected])[::-1]]
        hotspots=[{'triangle':int(i),'maximum_absolute_principal_strain':float(magnitude[i]),
                   'principal_strains':principal[i].tolist(),
                   'material_centre':self.material_points[faces[i]].mean(0).tolist(),
                   'state_centre':points[faces[i]].mean(0).tolist()} for i in selected]
        return {'geometry_valid':bool(np.isfinite(principal).all() and np.isfinite(areas).all()
                                      and (areas>np.finfo(float).eps*self.rest_area).all()),
                'pass': bool(maximum <= self.limits['max_edge_strain'] and membrane_max <= self.limits['max_edge_strain']
                             and abs(change) <= self.limits['max_area_change']),
                'maximum_absolute_edge_strain': maximum, 'median_absolute_edge_strain': float(np.median(abs(strain))),
                'maximum_principal_material_strain': membrane_max,
                'strain_hotspots':hotspots,
                'surface_topology':self.topology,
                'hotspot_scope':'Triangle/material coordinates and supplied state coordinates only; contact role, stress, yield and material parameters are not identified',
                'measurement': 'Actual triangle edges and principal in-plane stretches; auxiliary spring chords are not material lengths',
                'p95_absolute_edge_strain': float(np.percentile(abs(strain), 95)),
                'area_m2': area*self.unit_length_m**2, 'rest_area_m2': self.rest_area*self.unit_length_m**2, 'relative_area_change': change,
                'coordinates_unit_length_m':self.unit_length_m,
                'limits': dict(self.limits),
                'scope': 'Actual triangle surface versus declared intrinsic material metric; no ground-truth dimensions'}
