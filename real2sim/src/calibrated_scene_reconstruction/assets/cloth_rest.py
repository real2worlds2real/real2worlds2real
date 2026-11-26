"""Explicit intrinsic material coordinates for initially folded particle sheets."""
import numpy as np

KINDS = ('stretch', 'shear', 'bend')


def triangle_edges(faces):
    faces = np.asarray(faces)
    return np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)


def material_inverse(rest, faces):
    """Inverse rest-chart edge matrices, independent of world orientation."""
    tri = np.asarray(rest, float)[faces]
    a, b = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    x = np.linalg.norm(a, axis=1)
    y = np.sum(a*b, axis=1)/x
    z = np.linalg.norm(np.cross(a, b), axis=1)/x
    if not np.isfinite([x, y, z]).all() or min(x.min(), z.min()) <= 1e-12:
        raise ValueError('Nondegenerate intrinsic triangle chart required')
    dm = np.zeros((len(tri), 2, 2))
    dm[:, 0, 0] = x; dm[:, 0, 1] = y; dm[:, 1, 1] = z
    return np.linalg.inv(dm)


def principal_strains(points, faces, inverse):
    points = np.asarray(points, float)
    ds = np.stack([points[faces[:, 1]]-points[faces[:, 0]], points[faces[:, 2]]-points[faces[:, 0]]], axis=2)
    deformation = ds@inverse
    metric = np.swapaxes(deformation, 1, 2)@deformation
    return np.sqrt(np.maximum(np.linalg.eigvalsh(metric), 0.))-1


def triangulate_grid(points, rest, rows, columns, *, faces=None):
    """Choose quad diagonals that represent a folded sheet with less distortion.

    Only connectivity changes. Material coordinates, masses, dimensions and all
    vertex positions remain unchanged. This cannot remove true planar shear;
    new native dynamics, self-contact and image checks remain required.
    Pass saved ``faces`` when resuming a geometry proposal. Re-selecting the
    diagonals after moving vertices changes the actual surface and invalidates
    previous thickness/contact checks, even with identical vertex positions.
    """
    points, rest = np.asarray(points, float), np.asarray(rest, float)
    if (type(rows) is not int or type(columns) is not int or min(rows, columns) < 3
            or points.shape != (rows*columns, 3) or rest.shape != points.shape
            or not np.isfinite(points).all() or not np.isfinite(rest).all()):
        raise ValueError('Matching finite structured cloth states and grid dimensions required')
    grid = np.arange(rows*columns).reshape(rows, columns)
    chart = rest.reshape(rows, columns, 3)
    dx, dy = chart[0, -1]-chart[0, 0], chart[-1, 0]-chart[0, 0]
    expected = chart[:1, :, :]+chart[:, :1, :]-chart[0, 0]
    # Fold creases can need locally finer material samples. Connectivity changes
    # must preserve that explicit chart, including its nonuniform area weights.
    if (not np.allclose(chart, expected, atol=1e-10, rtol=1e-9)
            or np.linalg.norm(np.cross(dx, dy)) <= 1e-12
            or np.any(np.diff(chart[0], axis=0)@dx <= 0)
            or np.any(np.diff(chart[:, 0], axis=0)@dy <= 0)
            or not np.allclose(np.cross(chart[0]-chart[0, 0], dx), 0, atol=1e-10)
            or not np.allclose(np.cross(chart[:, 0]-chart[0, 0], dy), 0, atol=1e-10)):
        raise ValueError('Rest coordinates must describe a monotonic rectilinear material grid')
    a = grid[:-1, :-1].ravel()
    first = np.stack([np.c_[a, a+1, a+columns+1], np.c_[a, a+columns+1, a+columns]], axis=1)
    second = np.stack([np.c_[a, a+1, a+columns], np.c_[a+1, a+columns+1, a+columns]], axis=1)
    scores = []
    for candidate_faces in (first, second):
        flat = candidate_faces.reshape(-1, 3)
        values = principal_strains(points, flat, material_inverse(rest, flat))
        scores.append(abs(values).reshape(len(a), 2, 2).max(axis=(1, 2)))
    preserve = faces is not None
    if preserve:
        saved = np.asarray(faces)
        if saved.shape != (2*len(a), 3) or saved.dtype.kind not in 'iu':
            raise ValueError('Saved grid faces must retain the canonical per-cell topology')
        saved = saved.reshape(-1, 2, 3)
        use_first = np.all(saved == first, axis=(1, 2))
        use_second = np.all(saved == second, axis=(1, 2))
        if not np.all(use_first | use_second):
            raise ValueError('Saved grid faces must retain the canonical per-cell topology and winding')
        flip = use_second
    else:
        flip = scores[1]+1e-10 < scores[0]
    faces = np.where(flip[:, None, None], second, first).reshape(-1, 3)
    stretch = np.concatenate([np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()],
                              np.c_[grid[:-1].ravel(), grid[1:].ravel()]])
    positive, negative = np.c_[a, a+columns+1], np.c_[a+1, a+columns]
    shear = np.where(flip[:, None], negative, positive)
    opposite = np.where(flip[:, None], positive, negative)
    bend = np.concatenate([opposite, np.c_[grid[:, :-2].ravel(), grid[:, 2:].ravel()],
                           np.c_[grid[:-2].ravel(), grid[2:].ravel()]])
    material = {'points': rest.tolist(), 'springs': {k: v.tolist() for k, v in
                [('stretch', stretch), ('shear', shear), ('bend', bend)]}}
    validate(material, points, faces)
    return faces, material, {'quad_cells': len(a), 'flipped_diagonals': int(flip.sum()),
                             'old_maximum_principal_strain': float(scores[0].max()),
                             'new_maximum_principal_strain': float(np.where(flip, scores[1], scores[0]).max()),
                             'topology_preserved': preserve,
                             'vertex_positions_changed': False, 'rest_dimensions_changed': False,
                             'scope': 'Initial model meshing only, not task validation or runtime remeshing'}


def area(points, faces):
    points, faces = np.asarray(points, float), np.asarray(faces)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError('Finite material points and valid triangle indices required')
    values = np.linalg.norm(np.cross(points[faces[:,1]]-points[faces[:,0]],
                                     points[faces[:,2]]-points[faces[:,0]]),axis=1)/2
    if not np.isfinite(values).all() or values.min() <= 1e-14:
        raise ValueError('Nonfinite or degenerate cloth material area')
    return float(values.sum())


def validate(value, vertices, faces):
    if (not isinstance(value, dict) or not {'points', 'springs'}.issubset(value)
            or set(value)-{'points', 'springs', 'bend_memory', 'bend_reference', 'bend_angles', 'spring_model','components'}):
        raise ValueError('Cloth rest data requires points and typed springs')
    rest = np.asarray(value['points'], float)
    if rest.shape != np.asarray(vertices).shape or not np.isfinite(rest).all():
        raise ValueError('Cloth rest points must match finite simulation vertices')
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(rest)):
        raise ValueError('Cloth rest triangles must reference valid simulation vertices')
    areas = np.linalg.norm(np.cross(rest[faces[:, 1]]-rest[faces[:, 0]],
                                    rest[faces[:, 2]]-rest[faces[:, 0]]), axis=1)/2
    if not np.isfinite(areas).all() or np.min(areas) <= 1e-14:
        raise ValueError('Degenerate intrinsic cloth triangle')
    if not isinstance(value['springs'], dict) or set(value['springs']) != set(KINDS):
        raise ValueError('Declare stretch, shear and bend material springs explicitly')
    # Encode undirected pairs once. Per-spring Python sets and norm calls made
    # resolved cloth checks dominate each native startup; these array checks
    # preserve duplicate, topology, connectivity and explicit-island rejection.
    def codes(pairs):
        pairs=np.sort(np.asarray(pairs,dtype=np.int64),axis=1)
        return pairs[:,0]*len(rest)+pairs[:,1]
    material_edges=codes(triangle_edges(faces))
    lattice=None
    if 'spring_model' in value:
        if value['spring_model'] not in {'balanced_square_lattice','triangle_surface'}:
            raise ValueError('Unknown explicit cloth spring model')
        from .particle_lattice import topology
        expected,_=topology(rest,faces,model=value['spring_model'])
        lattice={kind:np.sort(codes(expected[kind])) for kind in KINDS}
    groups={};encoded=[];edges=[]
    for kind in KINDS:
        pairs=np.asarray(value['springs'][kind])
        if (pairs.ndim!=2 or pairs.shape[1:]!=(2,) or not len(pairs) or pairs.dtype.kind not in 'iu'
                or pairs.min()<0 or pairs.max()>=len(rest) or np.any(pairs[:,0]==pairs[:,1])):
            raise ValueError('Cloth springs require distinct valid integer vertex pairs')
        lengths=np.linalg.norm(rest[pairs[:,0]]-rest[pairs[:,1]],axis=1)
        code=codes(pairs)
        if not np.isfinite(lengths).all() or (lengths<=1e-10).any():
            raise ValueError('Duplicate or zero-rest-length cloth spring')
        if kind!='bend':
            allowed=np.isin(code,material_edges)
            if lattice is not None:allowed |= np.isin(code,lattice[kind])
            if not allowed.all():
                raise ValueError('Membrane springs must follow triangle material edges; a cross-face chord is a bending constraint')
            edges.append(pairs)
        if lattice is not None and not np.array_equal(np.unique(code),lattice[kind]):
            raise ValueError('Balanced lattice springs must match the existing material cells exactly')
        groups[kind]=pairs;encoded.append(code)
    all_codes=np.concatenate(encoded)
    if len(np.unique(all_codes))!=len(all_codes):
        raise ValueError('Duplicate or zero-rest-length cloth spring')
    membrane=np.concatenate(edges)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        # Blender's bundled Python has NumPy but need not include SciPy.
        # Keep the same graph test there, without adding a runtime dependency.
        neighbors=[[] for _ in rest]
        for a,b in membrane:
            neighbors[a].append(b);neighbors[b].append(a)
        labels=np.full(len(rest),-1,dtype=int);count=0
        for seed in range(len(rest)):
            if labels[seed]>=0:continue
            labels[seed]=count;pending=[seed]
            while pending:
                for node in neighbors[pending.pop()]:
                    if labels[node]<0:labels[node]=count;pending.append(node)
            count+=1
    else:
        graph=coo_matrix((np.ones(len(membrane),dtype=np.uint8),membrane.T),shape=(len(rest),len(rest))).tocsr()
        count,labels=connected_components(graph,directed=False)
    if count!=1 or 'components' in value:
        components=value.get('components')
        if not isinstance(components,list) or not components:
            raise ValueError('Material springs must connect the entire cloth surface')
        owners=np.full(len(rest),-1,dtype=int)
        sizes=np.bincount(labels)
        for index,component in enumerate(components):
            if (not isinstance(component,list) or len(component)<3 or len(set(component))!=len(component)
                    or any(type(i) is not int or not 0<=i<len(rest) for i in component)
                    or (owners[component]>=0).any()):
                raise ValueError('Explicit FEM islands must partition actual material vertices')
            ids=labels[component]
            if (ids!=ids[0]).any() or sizes[ids[0]]!=len(component):
                raise ValueError('Each FEM island must remain physically connected')
            owners[component]=index
        all_pairs=np.concatenate(list(groups.values()))
        if (owners<0).any() or (owners[all_pairs[:,0]]!=owners[all_pairs[:,1]]).any():
            raise ValueError('FEM islands cannot invent springs through empty space')
    reference=value.get('bend_reference')
    if reference is not None:
        if (not isinstance(reference,dict) or set(reference)!={'points','source','basis'}
                or any(not isinstance(reference.get(k),str) or not reference[k].strip() for k in ('source','basis'))):
            raise ValueError('Natural bend reference needs explicit points, source and construction basis')
        shape=np.asarray(reference['points'],float)
        if shape.shape!=rest.shape or not np.isfinite(shape).all():
            raise ValueError('Natural curvature reference must preserve material node correspondence')
        strains=principal_strains(shape,faces,material_inverse(rest,faces))
        if abs(strains).max()>1e-5:
            raise ValueError('Natural bend reference must preserve the intrinsic membrane metric')
    if 'bend_angles' in value:
        from .cloth_curvature import validate as validate_curvature
        validate_curvature(value['bend_angles'],faces)
    memory = value.get('bend_memory')
    if memory is not None:
        if (not isinstance(memory, dict) or set(memory) != {'indices', 'lengths', 'type', 'basis', 'source'}
                or memory['type'] != 'retained_crease'
                or any(not isinstance(memory.get(k), str) or not memory[k].strip() for k in ('basis', 'source'))):
            raise ValueError('Retained crease bending needs explicit source and construction basis')
        indices, lengths = np.asarray(memory['indices']), np.asarray(memory['lengths'], float)
        pairs = np.asarray(value['springs']['bend'], int)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                or indices.min() < 0 or indices.max() >= len(pairs) or len(np.unique(indices)) != len(indices)
                or lengths.shape != indices.shape or not np.isfinite(lengths).all() or lengths.min() <= 1e-10):
            raise ValueError('Retained crease lengths must reference distinct bending springs')
        intrinsic = np.linalg.norm(rest[pairs[indices, 0]]-rest[pairs[indices, 1]], axis=1)
        if np.any(lengths > intrinsic*(1+1e-6)):
            raise ValueError('Crease bending cannot expand intrinsic membrane dimensions')
    return {'rest_area_m2': float(areas.sum()), 'vertices': len(rest), 'springs': len(all_codes)}


def constraints(value, settings):
    """Membrane lengths stay intrinsic; explicit crease memory affects bends only.

    This is a distance-based bending approximation, not signed dihedral FEM.
    It can preserve an authored crease but must still pass native source-shape,
    deformation/contact and task checks; it never pins a material node.
    """
    rest = np.asarray(value['points'], float)
    indices, kinds = [], []
    for kind in KINDS:
        indices.extend(value['springs'][kind])
        kinds.extend([kind]*len(value['springs'][kind]))
    indices = np.asarray(indices, np.int32)
    lengths = np.linalg.norm(rest[indices[:, 0]]-rest[indices[:, 1]], axis=1)
    if value.get('bend_memory') is not None:
        memory = value['bend_memory']
        offset = len(value['springs']['stretch'])+len(value['springs']['shear'])
        lengths[offset+np.asarray(memory['indices'], int)] = memory['lengths']
    stiffness = [float(settings[kind+'_stiffness']) for kind in kinds]
    return indices, lengths, stiffness, kinds


def project_shape(value, vertices, faces, *, iterations=1500, reference_pull=.0001, collision_projection=None,device=None):
    """Regularize a draft shape toward its intrinsic edge lengths.

    An optional agent-authored surface projection can exclude known rigid
    surfaces. This is geometry initialization, not native physics, contact
    validation or task evidence. It does not resolve self-intersections.
    """
    validate(value, vertices, faces)
    if (type(iterations) is not int or not 1 <= iterations <= 20000
            or not np.isfinite(reference_pull) or not 0 <= reference_pull <= .1):
        raise ValueError('Bounded projection iterations and finite reference pull required')
    target = np.asarray(vertices, float)
    if not np.isfinite(target).all():
        raise ValueError('Cannot project nonfinite cloth positions')
    points = target.copy()
    pairs = np.concatenate([np.asarray(value['springs'][kind], int) for kind in ('stretch', 'shear')])
    a,b = pairs.T
    rest = np.asarray(value['points'], float)
    lengths = np.linalg.norm(rest[a]-rest[b], axis=1)
    # Resolve small fold edges as well as coarse straight strips: metre error
    # alone can hide very large relative strain on a tiny edge.
    weights = np.clip(np.median(lengths)/lengths, .1, 100.)
    degree = np.bincount(np.r_[a,b], weights=np.r_[weights,weights], minlength=len(points))
    degree = np.maximum(degree, 1)[:, None]
    if device is not None:
        import torch
        if not str(device).startswith('cuda:') or not torch.cuda.is_available():
            raise ValueError('Explicit available CUDA device required for geometry preparation')
        p=torch.as_tensor(points,dtype=torch.float64,device=device).clone()
        target_gpu=p.clone();aa=torch.as_tensor(a,device=device);bb=torch.as_tensor(b,device=device)
        rest_lengths=torch.as_tensor(lengths,device=device);w=torch.as_tensor(weights,device=device)
        d=torch.as_tensor(degree,device=device);projections=0
        for step in range(iterations):
            delta=p[bb]-p[aa];current=torch.linalg.vector_norm(delta,dim=1)
            correction=.5*(1-rest_lengths/current.clamp_min(1e-12))[:,None]*delta*w[:,None]
            move=torch.zeros_like(p);move.index_add_(0,aa,correction);move.index_add_(0,bb,-correction)
            p+=move/d+reference_pull*(target_gpu-p)
            if collision_projection is not None and ((step+1)%25==0 or step+1==iterations):
                projected=np.asarray(collision_projection(p.cpu().numpy()),float)
                if projected.shape!=points.shape or not np.isfinite(projected).all():
                    raise ValueError('Surface projection returned invalid cloth geometry')
                p.copy_(torch.as_tensor(projected,device=device));projections+=1
        points=p.cpu().numpy()
        strain=np.linalg.norm(points[b]-points[a],axis=1)/lengths-1
        return points,{'iterations':iterations,'reference_pull':float(reference_pull),'max_edge_strain':float(abs(strain).max()),
            'max_vertex_displacement_m':float(np.linalg.norm(points-target,axis=1).max()),'surface_projection_passes':projections,
            'device':str(device),'scope':'Geometry regularization only; no native state writes or task approval'}
    projections = 0
    for step in range(iterations):
        delta = points[b]-points[a]
        current = np.linalg.norm(delta, axis=1)
        correction = .5*(1-lengths/np.maximum(current, 1e-12))[:, None]*delta*weights[:, None]
        move = np.stack([np.bincount(a, weights=correction[:,k], minlength=len(points))
                         -np.bincount(b, weights=correction[:,k], minlength=len(points)) for k in range(3)], axis=1)
        points += move/degree+reference_pull*(target-points)
        if collision_projection is not None and ((step+1)%25==0 or step+1==iterations):
            projected=np.asarray(collision_projection(points),float)
            if projected.shape!=points.shape or not np.isfinite(projected).all():
                raise ValueError('Surface projection returned invalid cloth geometry')
            points=projected.copy();projections+=1
    strain = np.linalg.norm(points[b]-points[a], axis=1)/lengths-1
    return points, {'iterations': iterations, 'reference_pull': float(reference_pull),
                    'max_edge_strain': float(np.max(abs(strain))),
                    'max_vertex_displacement_m': float(np.linalg.norm(points-target, axis=1).max()),
                    'surface_projection_passes':projections,
                    'scope': 'Geometry regularization only; native contact/cloth task checks still required'}


def surface_chart(points, faces, *, source, basis):
    """Intrinsic chart for a connected native FEM shell construction hypothesis.

    Edge categories preserve legacy topology bookkeeping; FEM uses triangle
    continuum parameters, never these edges as calibrated particle springs.
    """
    points=np.asarray(points,float);faces=np.asarray(faces,int);edges=triangle_edges(faces)
    neighbors=[[] for _ in points]
    for a,b in edges:neighbors[a].append(b);neighbors[b].append(a)
    visited=set();stretch=[];components=[]
    for seed in range(len(points)):
        if seed in visited:continue
        component=[];pending=[seed];visited.add(seed)
        while pending:
            a=pending.pop();component.append(a)
            for b in neighbors[a]:
                if b not in visited:visited.add(b);pending.append(b);stretch.append(tuple(sorted((a,b))))
        components.append(sorted(component))
    spanning=set(stretch);shear=[tuple(e) for e in edges if tuple(e) not in spanning]
    from .cloth_curvature import adjacency
    pairs,_=adjacency(faces);edge_set={tuple(e) for e in edges};bend=set()
    for left,right in pairs:
        opposite=set(faces[left])^set(faces[right]);pair=tuple(sorted(opposite))
        if len(pair)==2 and pair not in edge_set:bend.add(pair)
    if not shear or not bend:raise ValueError('Flexible patch needs resolved membrane and bending topology')
    value={'points':points.tolist(),'springs':{k:np.asarray(v,dtype=int).tolist() for k,v in {'stretch':stretch,'shear':shear,'bend':sorted(bend)}.items()},
           'bend_reference':{'points':points.tolist(),'source':source,'basis':basis}}
    if len(components)>1:value['components']=[[int(i) for i in c] for c in components]
    validate(value,points,faces)
    return value
