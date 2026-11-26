"""Small rigid translations to prepare a nonpenetrating initial candidate.

This never edits a running simulation or changes shape, mass, friction or
constraints. Native stability, source scope and task checks remain mandatory.
"""
import numpy as np
from scipy.optimize import minimize


def separate(parts, movable, *, clearance_m, maximum_translation_m,
             spacing_m=.0005, device='cpu', iterations=30):
    from calibrated_scene_reconstruction.assets.rigid_initial_state import surface_samples
    from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface
    names = [p['name'] for p in parts]
    if (len(set(names)) != len(names) or not movable or not set(movable) <= set(names)
            or not np.isfinite([clearance_m, maximum_translation_m, spacing_m]).all()
            or not 0 < clearance_m < maximum_translation_m or spacing_m <= 0
            or type(iterations) is not int or not 1 <= iterations <= 100):
        raise ValueError('Unique parts, movable subset and bounded positive clearance required')
    free = [names.index(n) for n in names if n in movable]
    column = {i: 3*j for j, i in enumerate(free)}
    samples = [surface_samples(p['V'], p['F'], spacing_m) for p in parts]
    surfaces = [SignedSurface(p['V'], p['F'], device=device) for p in parts]
    translations = np.zeros((len(parts), 3)); trace = []
    unit = maximum_translation_m
    success = False
    for iteration in range(iterations):
        rows, bounds = [], []; worst = 0.
        for i in range(len(parts)):
            for j in range(len(parts)):
                if i == j:
                    continue
                points = samples[i]+translations[i]-translations[j]
                box = surfaces[j].bounds
                near = ((points >= box[0]-clearance_m) & (points <= box[1]+clearance_m)).all(1)
                if not near.any():
                    continue
                points = points[near]; distances = surfaces[j](points)
                selected = np.flatnonzero(distances < clearance_m-2e-6)
                if not len(selected):
                    continue
                worst = max(worst, float(clearance_m-distances[selected].min()))
                if distances[selected].min() < -1e-6:
                    # At a shared edge, the closest face is nonunique. Resolve
                    # the deepest actual overlap first, then requery every
                    # point; opposing edge normals cannot certify a step.
                    minimum = float(distances[selected].min())
                    selected = selected[distances[selected] <= minimum+max(clearance_m*.25, abs(minimum)*.1)]
                selected = selected[np.argsort(distances[selected])[:24]]
                corrections = surfaces[j].project_out(points[selected], clearance_m)-points[selected]
                for correction in corrections:
                    length = np.linalg.norm(correction)
                    if length <= 1e-9:
                        continue
                    row = np.zeros(3*len(free)); normal = correction/length
                    for index, sign in [(i, 1), (j, -1)]:
                        if index in column:
                            row[column[index]:column[index]+3] = sign*normal
                    if not row.any():
                        raise ValueError('Fixed parts interpenetrate or violate requested clearance')
                    rows.append(row)
                    bounds.append(length+normal@(translations[i]-translations[j]))
        trace.append({'iteration': iteration, 'maximum_clearance_deficit_m': worst})
        if not rows:
            success = True
            break
        A, b = np.asarray(rows), np.asarray(bounds)/unit
        initial = translations[free].reshape(-1)/unit
        result = minimize(lambda x: .5*float(x@x), initial, jac=lambda x: x,
                          constraints={'type': 'ineq', 'fun': lambda x: A@x-b, 'jac': lambda x: A},
                          bounds=[(-1., 1.)]*len(initial), method='SLSQP',
                          options={'maxiter': 100, 'ftol': 1e-12})
        trace[-1]['optimizer_success'] = bool(result.success)
        if not result.success:
            break
        translations[free] = result.x.reshape(-1, 3)*unit
    within_budget = all(np.linalg.norm(translations[i]) <= maximum_translation_m+1e-9 for i in free)
    return {'proposal_only': True, 'pass': bool(success and within_budget),
            'translations_m': dict(zip(names, translations.tolist())),
            'clearance_m': clearance_m, 'maximum_translation_m': maximum_translation_m,
            'trace': trace, 'scope': 'Finite sampled rigid translations only; rebuild and rerun full initial/native/source/task checks'}
