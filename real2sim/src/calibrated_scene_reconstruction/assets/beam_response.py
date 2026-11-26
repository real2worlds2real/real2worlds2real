"""Conditional small-load beam references for reviewing rigid approximations.

Inputs are measured geometry and declared material/support hypotheses. This
does not identify a real material, prove native contacts or approve an asset.
The caller must verify the assumed supports against actual native contacts.
"""
import numpy as np


def circular_section(outer_radius, inner_radius=0.):
    if (not np.isfinite([outer_radius, inner_radius]).all()
            or not 0 <= inner_radius < outer_radius):
        raise ValueError('Finite positive annular section required')
    return float(np.pi*(outer_radius**4-inner_radius**4)/4)


def section_moments(vertices, faces, origin, normal, offsets):
    """Oriented closed-solid section area and least centroidal second moment.

    Triangle/plane segments retain the source winding, including cavity loops.
    This is a finite geometric section measurement, not material inference.
    """
    v=np.asarray(vertices,float);f=np.asarray(faces,int)
    origin=np.asarray(origin,float);normal=np.asarray(normal,float);offsets=np.asarray(offsets,float)
    if (v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3
            or origin.shape!=(3,) or normal.shape!=(3,) or offsets.ndim!=1
            or not np.isfinite(v).all() or not np.isfinite(origin).all()
            or not np.isfinite(normal).all() or not np.isfinite(offsets).all()
            or not np.isclose(normal@normal,1.)):
        raise ValueError('Finite source triangles and unit section axis required')
    u=np.cross(normal,np.eye(3)[np.argmin(abs(normal))]);u/=np.linalg.norm(u)
    w=np.cross(normal,u);tri=v[f]-origin
    coordinate=tri@normal;face_normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    tangent=np.cross(normal,face_normal);rows=[]
    for offset in offsets:
        d=coordinate-offset;next_d=np.roll(d,-1,axis=1)
        cross=(d>0)!=(next_d>0);selected=np.flatnonzero(cross.sum(1)==2)
        edge=np.argwhere(cross[selected]);a=tri[selected[edge[:,0]],edge[:,1]]
        b=tri[selected[edge[:,0]],(edge[:,1]+1)%3]
        fraction=d[selected[edge[:,0]],edge[:,1]]/(d[selected[edge[:,0]],edge[:,1]]-next_d[selected[edge[:,0]],edge[:,1]])
        cut=(a+fraction[:,None]*(b-a)).reshape(-1,2,3)
        flip=((cut[:,1]-cut[:,0])*tangent[selected]).sum(1)<0;cut[flip]=cut[flip,::-1]
        x=cut@u;y=cut@w;c=x[:,0]*y[:,1]-x[:,1]*y[:,0];area=c.sum()/2
        if not np.isfinite(area) or area<=1e-16:
            raise ValueError('Beam section is empty, inverted or not a closed positive solid')
        cx=((x[:,0]+x[:,1])*c).sum()/(6*area);cy=((y[:,0]+y[:,1])*c).sum()/(6*area)
        ix=((y[:,0]**2+y[:,0]*y[:,1]+y[:,1]**2)*c).sum()/12-area*cy*cy
        iy=((x[:,0]**2+x[:,0]*x[:,1]+x[:,1]**2)*c).sum()/12-area*cx*cx
        ixy=((2*x[:,0]*y[:,0]+x[:,0]*y[:,1]+x[:,1]*y[:,0]+2*x[:,1]*y[:,1])*c).sum()/24-area*cx*cy
        least=float(np.linalg.eigvalsh([[ix,ixy],[ixy,iy]]).min())
        if not np.isfinite(least) or least<=0:raise ValueError('Positive section inertia required')
        rows.append({'coordinate_m':float(offset),'area_m2':float(area),'least_second_moment_m4':least})
    return rows


def nonuniform_tip_load(length, force, youngs_modulus, second_moments,
                       *, areas=None,poissons_ratio=.45,shear_coefficient=.5):
    """Mid-cell section reference for tip displacement of a clamped beam."""
    I=np.asarray(second_moments,float)
    if (I.ndim!=1 or len(I)<8 or not np.isfinite(I).all() or (I<=0).any()
            or not np.isfinite([length,force,youngs_modulus]).all()
            or min(length,youngs_modulus)<=0):
        raise ValueError('Resolved positive section profile and material modulus required')
    edges=np.linspace(0.,length,len(I)+1)
    weights=((length-edges[:-1])**3-(length-edges[1:])**3)/3
    bending=float(abs(force)*np.sum(weights/I)/youngs_modulus);shear=0.
    if areas is not None:
        A=np.asarray(areas,float)
        if (A.shape!=I.shape or not np.isfinite(A).all() or (A<=0).any()
                or not np.isfinite([poissons_ratio,shear_coefficient]).all()
                or not -1<poissons_ratio<.5 or not 0<shear_coefficient<=1):
            raise ValueError('Positive matching section areas and physical shear parameters required')
        G=youngs_modulus/(2*(1+poissons_ratio))
        shear=float(abs(force)*length/len(A)*np.sum(1/A)/(shear_coefficient*G))
    delta=bending+shear
    return {'maximum_deflection_m':delta,'deflection_over_span':delta/length,
            'bending_deflection_m':bending,'shear_deflection_m':shear,
            'material_identified':False,'supports_verified':False,
            'scope':'Finite nonuniform Euler–Bernoulli section reference; check section refinement, supports and three-dimensional response separately.'}


def point_load(length, load_coordinate, force, youngs_modulus, second_moment,
               *, support_span=None, samples=2001):
    """Euler–Bernoulli response to a transverse point load.

    No support_span means a cantilever clamped at zero. Otherwise two simple
    supports are at zero and support_span, with a possible free overhang.
    Signed reactions are returned: a unilateral contact must actually support
    those reaction directions; the formula cannot create a fixture.
    """
    values=[length, load_coordinate, force, youngs_modulus, second_moment]
    if (not np.isfinite(values).all() or min(length, youngs_modulus, second_moment) <= 0
            or not 0 <= load_coordinate <= length or type(samples) is not int or samples < 101):
        raise ValueError('Finite positive beam geometry, rigidity and resolved sampling required')
    x=np.linspace(0.,length,samples);c=load_coordinate
    if support_span is None:
        y=np.where(x <= c, x*x*(3*c-x), c*c*(3*x-c))*force/6
        reactions={'fixed_force_N':float(force),'fixed_moment_Nm':float(force*c)}
    else:
        a=float(support_span)
        if not np.isfinite(a) or not 0<a<=length:
            raise ValueError('Two distinct support locations inside the beam required')
        r0=force*(a-c)/a;ra=force*c/a
        slope=-(r0*a**3-force*max(a-c,0.)**3)/(6*a)
        y=(r0*x**3+ra*np.maximum(x-a,0.)**3-force*np.maximum(x-c,0.)**3)/6+slope*x
        reactions={'at_zero_N':float(r0),'at_support_span_N':float(ra)}
    y/=youngs_modulus*second_moment
    worst=int(np.argmax(abs(y)))
    return {'maximum_deflection_m':float(abs(y[worst])),
            'deflection_over_span':float(abs(y[worst])/length),
            'worst_coordinate_m':float(x[worst]),'tip_deflection_m':float(y[-1]),
            'reactions':reactions,'samples':samples,
            'material_identified':False,'supports_verified':False,
            'scope':'Conditional linear beam reference only; native support, shear, geometry and task checks remain separate.'}
