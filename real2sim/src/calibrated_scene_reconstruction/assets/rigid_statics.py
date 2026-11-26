"""Necessary force-balance diagnostic for an explicitly hypothesized contact set.

No contacts, supports or fixtures are invented here. A feasible static solution
does not establish contact geometry, compliance, stability or native behavior.
"""
import numpy as np
from scipy.optimize import linprog


def point_contact_balance(bodies, contacts, *, gravity=(0., 0., -9.81), cone_sides=32):
    """Nonadhesive point contacts with an inscribed polygonal friction cone.

    bodies maps names to mass and centre. Each contact supplies a, optional b
    (None means a fixed environment), point, normal (force direction on a), and
    friction. Equal and opposite contact forces act on b. No contact moments,
    preloads, springs, pose locks or externally applied holding forces exist.
    """
    g = np.asarray(gravity, float)
    if (not bodies or g.shape != (3,) or not np.isfinite(g).all() or np.linalg.norm(g) <= 0
            or type(cone_sides) is not int or not 8 <= cone_sides <= 128):
        raise ValueError('Bodies, finite nonzero gravity and 8..128 cone sides required')
    names = list(bodies); rows = {}; centres = {}; masses = {}
    for i, (name, body) in enumerate(bodies.items()):
        c = np.asarray(body['centre'], float); m = float(body['mass'])
        if c.shape != (3,) or not np.isfinite(c).all() or not np.isfinite(m) or m <= 0:
            raise ValueError('Finite body centres and positive masses required')
        rows[name] = 6*i; centres[name] = c; masses[name] = m
    columns = []; generators = []; parsed = []
    lengths = {name: 1e-6 for name in names}
    angles = np.arange(cone_sides)*2*np.pi/cone_sides
    for contact in contacts:
        a, b = contact['a'], contact.get('b')
        p, n = (np.asarray(contact[key], float) for key in ('point', 'normal'))
        mu = float(contact['friction'])
        if (a not in rows or (b is not None and b not in rows) or a == b
                or p.shape != (3,) or n.shape != (3,) or not np.isfinite(np.r_[p, n, mu]).all()
                or np.linalg.norm(n) < 1e-10 or mu < 0):
            raise ValueError('Valid contact bodies, point, normal and nonnegative friction required')
        n = n/np.linalg.norm(n)
        basis = np.linalg.svd(n[None], full_matrices=True)[2][1:]
        directions = n+mu*np.c_[np.cos(angles), np.sin(angles)]@basis
        block = np.zeros((len(names)*6, cone_sides))
        for name, sign in [(a, 1.), (b, -1.)]:
            if name is None:
                continue
            j = rows[name]; arm = p-centres[name]
            lengths[name] = max(lengths[name], float(np.linalg.norm(arm)))
            block[j:j+3] = sign*directions.T
            block[j+3:j+6] = sign*np.cross(arm, directions).T
        columns.append(block); generators.append(directions); parsed.append((a, b))
    rhs = np.zeros(len(names)*6); scales = np.ones_like(rhs)
    for name in names:
        i = rows[name]; weight = masses[name]*np.linalg.norm(g)
        rhs[i:i+3] = -masses[name]*g
        scales[i:i+3] = weight; scales[i+3:i+6] = weight*lengths[name]
    report = {'equilibrium_feasible': False, 'contact_forces_on_a': [],
              'cone_sides': cone_sides, 'contact_count': len(contacts),
              'diagnostic_only': True, 'task_approved': False,
              'scope': 'Necessary balance for supplied point contacts and friction priors only; inscribed cone approximation, not a native stability certificate'}
    if not columns:
        return {**report, 'reason': 'No hypothesized contacts balance gravity'}
    matrix = np.concatenate(columns, axis=1)
    solution = linprog(np.ones(matrix.shape[1]), A_eq=matrix/scales[:, None],
                       b_eq=rhs/scales, bounds=(0., None), method='highs')
    report['solver_status'] = int(solution.status)
    report['solver_message'] = solution.message
    if solution.success:
        residual = float(np.max(abs(matrix@solution.x-rhs)/scales))
        report['normalized_wrench_residual'] = residual
        report['equilibrium_feasible'] = residual <= 1e-6
        for i, ((a, b), directions) in enumerate(zip(parsed, generators)):
            force = solution.x[i*cone_sides:(i+1)*cone_sides]@directions
            report['contact_forces_on_a'].append({'a': a, 'b': b, 'force_N': force.tolist()})
    return report
