"""Actual-surface queries for inferred construction paths, never native proof."""
import numpy as np


def initial_targets(plan,parts):
    """CPU frame preflight for a uniform-mass surface-FEM construction seed.

    Source link vertices share the assembly frame. The actual native check is
    still mandatory after reset; future-start grips and other backends are not
    certified by this geometric preflight.
    """
    from .force_fixture import material_patch
    sheet=next(p for p in parts if p['name']==plan['cloth_link'])
    if sheet.get('backend')!='surface_fem':
        return {'applicable':False,'native_check_required':True,'reason':'Other native mass models retain runtime validation'}
    tolerance=plan.get('initial_target_tolerance_m')
    if tolerance is None:return {'applicable':False,'native_check_required':True,'reason':'No explicit preflight frame budget'}
    if not np.isfinite(tolerance) or not 0<tolerance<=.01:raise ValueError('Bounded initial target tolerance required')
    chart=material_coordinates(sheet['cloth_rest']['points'],plan.get('material_chart_rotation_rad',0.))
    rows=[]
    for grip in plan.get('grips',[]):
        if grip['active_s'][0]!=0:continue
        anchor=next(p for p in parts if p['name']==grip['anchor_link'])
        if anchor['kind']!='rigid':raise ValueError('Source preparation anchor must be rigid')
        ids=material_patch(chart,grip['centre_fraction'],grip['size_m'])
        expected=np.asarray(sheet['V'])[ids].mean(0)
        actual=np.asarray(grip['keyframes'],float)[0,1:]
        error=float(np.linalg.norm(actual-expected))
        rows.append({'anchor_link':grip['anchor_link'],'error_m':error,'pass':error<=tolerance})
    return {'applicable':bool(rows),'pass':all(row['pass'] for row in rows),'rows':rows,
            'tolerance_m':tolerance,'native_check_required':True,'task_approved':False}


def footprint_hull_placement(footprint,support,*,clearance_m=0.):
    """Translate an unchanged 2D construction footprint inside a support hull.

    This only screens outer-boundary fit. The hull does not fill collision
    holes or prove support/contact: actual triangle queries and native release
    remain required. In particular, it cannot approve a sheet over an outlet.
    """
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    p,s=np.asarray(footprint,float),np.asarray(support,float)
    if (p.ndim!=2 or s.ndim!=2 or p.shape[1:]!=(2,) or s.shape[1:]!=(2,)
            or min(len(p),len(s))<3 or not np.isfinite(p).all() or not np.isfinite(s).all()
            or not np.isfinite(clearance_m) or clearance_m<0
            or np.linalg.matrix_rank(p-p.mean(0),tol=1e-10)<2
            or np.linalg.matrix_rank(s-s.mean(0),tol=1e-10)<2):
        raise ValueError('Resolved finite planar footprints and nonnegative outline clearance required')
    equation=ConvexHull(s).equations
    normal,offset=equation[:,:2],equation[:,2]
    # The most exposed point in each support-normal direction determines
    # containment of the complete unchanged footprint, not just its centroid.
    extent=(p@normal.T).max(0)
    fit=linprog([0.,0.,-1.],A_ub=np.c_[normal,np.ones(len(normal))],
                b_ub=-offset-extent,bounds=[(None,None)]*3,method='highs')
    if not fit.success:raise ValueError('Bounded support-footprint placement could not be resolved')
    translation=fit.x[:2];margin=float((-((p+translation)@normal.T+offset)).min())
    return {'proposal_only':True,'task_approved':False,'contact_proven':False,
            'outer_boundary_fit':bool(margin>=clearance_m-1e-12),
            'translation_xy_m':translation.tolist(),'minimum_hull_margin_m':margin,
            'required_outline_clearance_m':float(clearance_m),
            'geometry_scaled':False,'collision_holes_filled':False,
            'scope':'Unchanged footprint versus projected outer hull only. Actual surface support, holes, material validity, native preparation and release remain unchecked.'}


def reanchor(plan,before,after,*,tolerance=1e-6):
    """Rebase tool paths after verified translation-only model centering.

    The dictionaries contain actual named rigid V/F meshes. This is not a fit
    of changed geometry: topology, indexed vertices and shape must agree.
    """
    from copy import deepcopy
    if not np.isfinite(tolerance) or not 0<tolerance<=1e-5:
        raise ValueError('Explicit small model-centering tolerance required')
    result=deepcopy(plan);offsets={}
    for grip in result.get('grips',[]):
        name=grip['anchor_link']
        if name not in offsets:
            if name not in before or name not in after:
                raise ValueError('Both actual rigid anchor meshes required')
            a,b=before[name],after[name]
            v,w=np.asarray(a['V'],float),np.asarray(b['V'],float)
            if (a['kind']!='rigid' or b['kind']!='rigid' or v.ndim!=2 or v.shape[1:]!=(3,)
                    or v.shape!=w.shape or not len(v) or not np.isfinite([v,w]).all()
                    or not np.array_equal(a['F'],b['F'])):
                raise ValueError('Unchanged indexed rigid anchor topology required for path rebasing')
            delta=(w.min(0)+w.max(0)-v.min(0)-v.max(0))/2
            if np.linalg.norm(v+delta-w,axis=1).max()>tolerance:
                raise ValueError('Rigid anchor changed beyond a verified centering translation')
            offsets[name]=delta
        path=np.asarray(grip['keyframes'],float)
        if path.ndim!=2 or path.shape[1:]!=(4,) or not np.isfinite(path).all():
            raise ValueError('Finite timed 3D tool path required for rebasing')
        path[:,1:]+=offsets[name];grip['keyframes']=path.tolist()
    result.pop('candidate_mesh_sha256',None)
    result.setdefault('initial_target_tolerance_m',1e-4)
    result['anchor_frame_rebasing']={'translation_m':{name:delta.tolist() for name,delta in offsets.items()},
        'indexed_geometry_tolerance_m':float(tolerance),'task_approved':False,
        'scope':'Actual rigid-mesh translation only; tool paths follow the new model frame. No geometry, force or task approval.'}
    return result


def material_coordinates(points,angle=0.):
    """A selection frame for a planar chart; never change the rest material."""
    points=np.asarray(points,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not len(points)
            or not np.isfinite(points).all() or not np.isfinite(angle) or abs(angle)>np.pi
            or np.ptp(points[:,2])>1e-9):
        raise ValueError('Finite planar material chart and selection-frame angle within pi required')
    c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return points@r.T


def nearby_patch(points,fraction,size,*,maximum_shift=.012,exclude=(),max_attempts=128):
    """Resolve a finite patch near clipped material boundaries, without filling holes.

    Only existing material nodes can be selected. A bounded proposal adjustment
    changes a temporary tool's location, never material quantity or native state.
    """
    from .force_fixture import material_patch
    points=np.asarray(points,float);fraction=np.asarray(fraction,float);size=np.asarray(size,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not len(points) or not np.isfinite(points).all()
            or fraction.shape!=(2,) or size.shape!=(2,) or not np.isfinite([fraction,size]).all()
            or (fraction<0).any() or (fraction>1).any() or (size<=0).any()
            or not np.isfinite(maximum_shift) or not 0<maximum_shift<=.05
            or type(max_attempts) is not int or not 1<=max_attempts<=256):
        raise ValueError('Finite chart, bounded material request, local shift and search budget required')
    low=points[:,:2].min(0);span=np.ptp(points[:,:2],axis=0)
    if (span<=0).any():raise ValueError('Resolved two-dimensional material required')
    wanted=low+fraction*span;distance=np.linalg.norm(points[:,:2]-wanted,axis=1)
    candidates=[wanted,*points[np.argsort(distance)[:max_attempts-1],:2]];avoid=set(exclude)
    for centre in candidates:
        shift=float(np.linalg.norm(centre-wanted))
        if shift>maximum_shift:continue
        selected=(centre-low)/span
        try:ids=material_patch(points,selected,size)
        except ValueError:continue
        if avoid.intersection(ids):continue
        return {'fraction':selected.tolist(),'nodes':ids,'requested_fraction':fraction.tolist(),
                'shift_m':shift,'proposal_only':True,'task_approved':False}
    raise ValueError('No disjoint resolved material patch within the bounded local search')


def spread_patches(points,rest,fractions,size_m,*,count=2,maximum_shift=.01,exclude=()):
    """Choose spatially separated, materially disjoint construction handles.

    Different material fractions can lie on coincident folded plies. These
    temporary support handles should span the intended posed region instead
    of accidentally supporting the same edge twice. The agent supplies the
    candidate role/region; this does not choose task pinch contacts or prove a
    grasp, native support or valid cloth geometry.
    """
    p,r,f=np.asarray(points,float),np.asarray(rest,float),np.asarray(fractions,float)
    if (p.shape!=r.shape or p.ndim!=2 or p.shape[1:]!=(3,) or not np.isfinite(p).all()
            or f.ndim!=2 or f.shape[1:]!=(2,) or not 2<=len(f)<=64
            or type(count) is not int or not 2<=count<=min(8,len(f))):
        raise ValueError('Matching posed/material points and bounded construction-handle candidates required')
    options=[]
    for fraction in f:
        try:item=nearby_patch(r,fraction,size_m,maximum_shift=maximum_shift,exclude=exclude)
        except ValueError:continue
        options.append({**item,'centre':p[item['nodes']].mean(0)})
    pairs=[(np.linalg.norm(a['centre']-b['centre']),i,j)for i,a in enumerate(options)
           for j,b in enumerate(options[:i])if not np.intersect1d(a['nodes'],b['nodes']).size]
    if not pairs:raise ValueError('No disjoint resolved construction handles')
    _,i,j=max(pairs);chosen=[i,j];owned=set(np.r_[options[i]['nodes'],options[j]['nodes']])
    while len(chosen)<count:
        available=[(min(np.linalg.norm(item['centre']-options[j]['centre'])for j in chosen),i)
                   for i,item in enumerate(options)if i not in chosen and not owned.intersection(item['nodes'])]
        if not available:raise ValueError('Cannot resolve the requested disjoint construction handles')
        _,i=max(available);chosen.append(i);owned.update(options[i]['nodes'])
    return {'patches':[options[i]for i in chosen],
            'minimum_spatial_separation_m':float(min(np.linalg.norm(options[i]['centre']-options[j]['centre'])
                for n,i in enumerate(chosen)for j in chosen[:n])),
            'proposal_only':True,'task_approved':False,'native_positions_written':False,
            'scope':'Spatially distributed preparation handles only; no task pinch or support approval.'}


def grip_budget(*,held_mass,flexural_rigidity,patch_width,bend_radius,
                youngs_modulus,thickness,tracking_error,tool_force_limit,
                allowable_strain=.01,load_factor=2.):
    """Dimensional load proposal for a temporary construction tool.

    D*w/r**2 estimates a bending force scale, not measured contact resistance.
    Geometry, friction and connected-sheet stress can invalidate this estimate;
    actual tracking and native material/contact checks remain mandatory.
    Neither this tool budget nor task success identifies material constants.
    """
    values=[held_mass,flexural_rigidity,patch_width,bend_radius,youngs_modulus,
            thickness,tracking_error,tool_force_limit,allowable_strain,load_factor]
    if (not np.isfinite(values).all() or min(values)<=0
            or allowable_strain>1 or load_factor<1):
        raise ValueError('Positive finite physical inputs, bounded strain and load factor at least one required')
    gravity=held_mass*9.81
    bending=flexural_rigidity*patch_width/bend_radius**2
    requested=gravity+load_factor*bending
    tensile=youngs_modulus*thickness*patch_width*allowable_strain
    limit=min(tool_force_limit,tensile)
    return {'proposal_only':True,'task_approved':False,'material_identified':False,
            'gravity_force_N':float(gravity),'bending_force_scale_N':float(bending),
            'requested_force_N':float(requested),'tool_force_limit_N':float(tool_force_limit),
            'tensile_force_scale_N':float(tensile),'force_limit_N':float(min(requested,limit)),
            'estimated_load_supported':bool(requested<=limit),
            'stiffness_N_m':float(min(requested,limit)/tracking_error),
            'tracking_error_target_m':float(tracking_error)}


def sliding_stations(rows, layer_offsets, turns, material_length):
    """Material stations for parallel offset curves with different arc lengths.

    This local packing proposal permits layers to shift longitudinally instead
    of demanding identical material stations around different bend radii.
    Connected-sheet shear, global isometry and contact still need independent
    actual-surface/native checks; this formula does not certify them.
    """
    rows,offsets,turns=np.asarray(rows,float),np.asarray(layer_offsets,float),np.asarray(turns,float)
    if (rows.ndim!=1 or len(rows)<2 or offsets.ndim!=1 or not len(offsets)
            or turns.shape!=rows.shape or not np.isfinite(rows).all() or not np.isfinite(offsets).all()
            or not np.isfinite(turns).all() or not np.isfinite(material_length) or material_length<=0
            or (np.diff(rows)<=0).any()):
        raise ValueError('Increasing centreline rows, finite layer offsets, turns and full material length required')
    result=rows[:,None]+(turns-turns[0])[:,None]*offsets[None]/material_length
    if (result<0).any() or (result>1).any() or (np.diff(result,axis=0)<=0).any():
        raise ValueError('Offset curves exceed the complete material or reverse its station order')
    return result


def transport_centres(reference, destination, rotation):
    """Move a resolved patch group without collapsing its distinct layers.

    A reference may be an explicitly inferred developable packing profile.
    This generates finite-controller goals only, never native positions or a
    rest-material edit. The full folded surface still needs independent tests.
    """
    points,centre,r=np.asarray(reference,float),np.asarray(destination,float),np.asarray(rotation,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<2 or centre.shape!=(3,)
            or r.shape!=(3,3) or not np.isfinite(points).all() or not np.isfinite(centre).all()
            or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-8)
            or not np.isclose(np.linalg.det(r),1.,atol=1e-8)):
        raise ValueError('Finite patch group, centre and proper unscaled transport rotation required')
    return (points-points.mean(0))@r.T+centre


def support_slide(points,parts,displacement_xy,*,maximum_lowering_m,clearance_m,top,samples=9):
    """Propose finite-controller centres along a real support, retaining holes.

    The agent supplies the feed direction and distance in an upright planning
    frame. Sample every supplied patch point against actual triangles;
    a partial insert's bounding box is never substituted for its missing half.
    These are preparation targets only, not cloth coordinates or a proof that
    the connected sheet can follow. Native motion, full-sheet contact and
    retention after all tools are released remain mandatory.
    """
    p,d=np.asarray(points,float),np.asarray(displacement_xy,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not 3<=len(p)<=4096
            or d.shape!=(2,) or not np.isfinite(p).all() or not np.isfinite(d).all()
            or np.linalg.norm(d)<=0 or np.linalg.matrix_rank(p-p.mean(0),tol=1e-10)<2
            or not np.isfinite([maximum_lowering_m,clearance_m,top]).all()
            or maximum_lowering_m<0 or clearance_m<=0 or type(samples) is not int
            or not 2<=samples<=128 or top<p[:,2].max()):
        raise ValueError('Resolved patch, finite feed, lowering/clearance and bounded support sampling required')
    centre=p.mean(0);offset=p-centre;goals=[];minimum_gap=float('inf')
    for fraction in np.linspace(0.,1.,samples):
        xy=p[:,:2]+fraction*d
        rays=downward_hits(parts,xy,top)['rays']
        if any(not ray['hit'] for ray in rays):
            raise ValueError('Feed footprint crosses a real support hole or boundary; choose another patch/path')
        height=np.asarray([ray['height_m'] for ray in rays])
        floor=float((height+clearance_m-offset[:,2]).max())
        if floor>centre[2]+1e-9:
            raise ValueError('Feed path requires lifting over an obstacle; a downward slide cannot certify it')
        z=max(centre[2]-fraction*maximum_lowering_m,floor)
        goals.append([*(centre[:2]+fraction*d),z])
        minimum_gap=min(minimum_gap,float((z+offset[:,2]-height).min()))
    return {'centres':goals,'fractions':np.linspace(0.,1.,samples).tolist(),
            'minimum_sampled_support_gap_m':minimum_gap,
            'sampled_footprints':samples,'patch_points':len(p),
            'support_holes_filled':False,'native_positions_written':False,
            'proposal_only':True,'task_approved':False,
            'scope':'Supplied patch points sampled over actual support triangles; continuous patch interiors are not certified. Finite preparation, full-sheet geometry and unforced retention still required.'}


def support_seating(points,parts,*,clearance_m,top,maximum_shift_m):
    """Seat a proposed section on actual selected support, retaining holes.

    Lower plies over a void do not set the supported ply's height. Return a
    bounded vertical target correction from the points with real support
    hits. Callers must regenerate the connected geometry and independently
    check whole-sheet routing, all solids, thickness and native release.
    """
    p=np.asarray(points,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,)or not 3<=len(p)<=4096 or not np.isfinite(p).all()
            or not np.isfinite([clearance_m,top,maximum_shift_m]).all()
            or min(clearance_m,maximum_shift_m)<=0 or top<p[:,2].max()):
        raise ValueError('Finite section, explicit support ray origin and bounded seating correction required')
    rays=downward_hits(parts,p[:,:2],top)['rays'];ids=np.array([i for i,r in enumerate(rays)if r['hit']],int)
    if not len(ids):raise ValueError('No actual support beneath the supplied section')
    heights=np.asarray([rays[i]['height_m']for i in ids])
    shift=float((heights+clearance_m-p[ids,2]).max())
    if abs(shift)>maximum_shift_m:raise ValueError('Actual support seating exceeds its explicit proposal budget')
    return {'vertical_shift_m':shift,'centre':(p.mean(0)+[0,0,shift]).tolist(),
            'supported_sample_fraction':len(ids)/len(p),'supported_samples':len(ids),'sample_count':len(p),
            'minimum_seated_support_gap_m':float((p[ids,2]+shift-heights).min()),
            'rays':rays,'support_holes_filled':False,'native_positions_written':False,
            'proposal_only':True,'task_approved':False,
            'scope':'Actual selected support under supplied points only; no continuous contact, whole-sheet geometry, load, retention or task approval.'}


def cavity_descent(points,parts,*,maximum_lowering_m,clearance_m):
    """Bound a vertical construction handle by its actual cavity surfaces.

    Query below each supplied posed patch point, including partial shelves
    and internal contents. A reported opening depth alone cannot certify free
    space. This produces a finite-controller target, never cloth coordinates
    or acceptance; full-sheet contact and released support remain mandatory.
    """
    p=np.asarray(points,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not 3<=len(p)<=4096
            or not np.isfinite(p).all()
            or not np.isfinite([maximum_lowering_m,clearance_m]).all()
            or min(maximum_lowering_m,clearance_m)<=0):
        raise ValueError('Resolved finite patch and positive bounded descent/clearance required')
    # Screen initial solid validity separately. Each ray starts at its own
    # point: a shared lower origin could skip a shelf beneath a higher point.
    hits=[downward_hits(parts,point[None,:2],float(point[2]))['rays'][0]for point in p]
    if any(not row['hit']for row in hits):
        raise ValueError('Unresolved cavity floor; do not infer available depth from an empty ray')
    floor=np.asarray([row['height_m']for row in hits])
    available=float((p[:,2]-floor-clearance_m).min())
    if available<=0:raise ValueError('Patch has no resolved downward clearance')
    distance=min(float(maximum_lowering_m),available)
    return {'centre':(p.mean(0)-[0,0,distance]).tolist(),'lowering_m':distance,
            'available_lowering_m':available,'requested_lowering_m':float(maximum_lowering_m),
            'clearance_m':float(clearance_m),'rays':hits,
            'proposal_only':True,'task_approved':False,'native_positions_written':False,
            'scope':'Sampled vertical patch-point clearance only. No full-sheet path, source, material or released-support approval.'}


def downward_hits(parts, xy, top):
    """First triangle below each local vertical ray, retaining holes and slopes.

    Parts and rays share an explicit upright planning frame. Callers select
    stationary solids: a moving lid's initial pose is not a cavity floor.
    Missing hits stay missing; bounds never substitute for a real surface.
    Finite-area material patches need multiple rays and fresh collision tests.
    """
    xy=np.asarray(xy,float)
    if (xy.ndim!=2 or xy.shape[1:]!=(2,) or not 1<=len(xy)<=4096
            or not np.isfinite(xy).all() or not np.isfinite(top) or not parts):
        raise ValueError('Finite bounded local rays, top height and actual solid triangles required')
    surfaces=[]
    for part in parts:
        v,f=np.asarray(part['V'],float),np.asarray(part['F'])
        if (not part.get('name') or v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all()
                or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or f.dtype.kind not in 'iu'
                or f.min()<0 or f.max()>=len(v)):
            raise ValueError('Named finite actual triangle surfaces required')
        a=v[f[:,0]];e=v[f[:,1]]-a;g=v[f[:,2]]-a
        determinant=e[:,0]*g[:,1]-e[:,1]*g[:,0]
        keep=abs(determinant)>1e-14
        surfaces.append((part['name'],np.flatnonzero(keep),a[keep],e[keep],g[keep],determinant[keep]))
    result=[]
    for point in xy:
        best=None
        for name,indices,a,e,g,det in surfaces:
            offset=point-a[:,:2]
            u=(offset[:,0]*g[:,1]-offset[:,1]*g[:,0])/det
            w=(e[:,0]*offset[:,1]-e[:,1]*offset[:,0])/det
            z=a[:,2]+u*e[:,2]+w*g[:,2]
            selected=(u>=-1e-10)&(w>=-1e-10)&(u+w<=1+1e-10)&(z<=top)
            if not selected.any():continue
            index=int(np.flatnonzero(selected)[np.argmax(z[selected])])
            if best is None or z[index]>best['height_m']:
                best={'height_m':float(z[index]),'part':name,'triangle':int(indices[index])}
        result.append({'xy':point.tolist(),'hit':best is not None,**({} if best is None else best)})
    return {'proposal_only':True,'native_validated':False,'task_approved':False,'rays':result}
