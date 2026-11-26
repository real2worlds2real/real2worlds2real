"""Diagnostic elastic-energy estimate; never a native constitutive certificate."""
import numpy as np

from .cloth_rest import material_inverse


def estimate(rest,points,faces,*,youngs_modulus,poissons_ratio,thickness,areal_density):
    """Plane-stress StVK estimate from the immutable intrinsic material metric.

    The equivalent speed is a whole-sheet energy scale, not a predicted native
    velocity. Gravity, friction, bending and contact may exchange/dissipate it.
    A small geometric error can store significant energy in a stiff thin sheet.
    """
    rest,points,faces=np.asarray(rest,float),np.asarray(points,float),np.asarray(faces)
    if (rest.shape!=points.shape or rest.ndim!=2 or rest.shape[1:]!=(3,)
            or not np.isfinite(rest).all() or not np.isfinite(points).all()
            or not np.isfinite([youngs_modulus,poissons_ratio,thickness,areal_density]).all()
            or min(youngs_modulus,thickness,areal_density)<=0 or not -1<poissons_ratio<.5):
        raise ValueError('Finite material/state coordinates and positive continuum estimates required')
    inverse=material_inverse(rest,faces)
    triangle=points[faces]
    deformation=np.stack([triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]],-1)@inverse
    strain=.5*(deformation.transpose(0,2,1)@deformation-np.eye(2))
    tri=rest[faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    lam=youngs_modulus*thickness*poissons_ratio/(1-poissons_ratio**2)
    mu=youngs_modulus*thickness/(2*(1+poissons_ratio))
    density=.5*lam*np.trace(strain,axis1=1,axis2=2)**2+mu*(strain*strain).sum((1,2))
    energy=float(density@area);mass=float(area.sum()*areal_density)
    return {'plane_stress_membrane_energy_estimate_J':energy,
            'whole_sheet_energy_speed_scale_m_s':float(np.sqrt(max(0.,2*energy/mass))),
            'rest_area_m2':float(area.sum()),'mass_kg':mass,
            'native_constitutive_readback':False,'physical_parameters_identified':False,
            'scope':'Diagnostic energy scale from declared continuum priors; not a new gate or native velocity prediction'}
