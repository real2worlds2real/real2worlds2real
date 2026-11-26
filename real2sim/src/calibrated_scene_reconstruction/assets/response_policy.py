"""Separate approximate flexible response from geometry and task acceptance.

Numerical strain targets remain raw diagnostics. They do not replace semantic,
contact, stability or actual task checks, and never relax rigid geometry.
"""
MODES={'task_and_semantics','quantitative'}


def mode(value=None):
    value='task_and_semantics' if value is None else value
    if value not in MODES:raise ValueError('Unknown flexible accuracy policy')
    return value


def material_verdict(measurement,accuracy_mode):
    accuracy_mode=mode(accuracy_mode)
    raw=measurement.get('pass') is True
    valid=measurement.get('geometry_valid') is True
    return {'pass':bool(valid and (raw or accuracy_mode=='task_and_semantics')),
            'accuracy_mode':accuracy_mode,'quantitative_accuracy_pass':raw,
            'geometry_valid':valid,'task_accepted':False,
            'scope':'Material-response accuracy only. Semantics, physical plausibility, contacts and the requested task must pass separately.'}


def approximate_part(part):
    """A flexible declaration cannot turn a rigid substitute into an exemption."""
    return (part.get('representation')=='deformable'
            and part.get('response_accuracy','task_and_semantics')=='task_and_semantics')


def secondary_contact_budgets(parts, understanding, accuracy_mode):
    """Bound secondary flexible details; final native semantic review is required.

    Primary task contacts and every rigid surface keep their original limits.
    Ten percent of the individual detail's span (at most2mm) bounds the
    qualitative approximation; it is not a measured material property.
    """
    import numpy as np
    if mode(accuracy_mode)!='task_and_semantics':return {}
    eligible={link for item in understanding['parts'] if approximate_part(item)
              and item.get('task_primary') is False for link in item['links']}
    return {p['name']:float(min(.002,.2*np.linalg.norm(
                np.asarray(p['V'])-np.asarray(p['V']).mean(axis=0),axis=1).max()))
            for p in parts if p['kind']=='cloth' and p['name'] in eligible}


def contact_limit(numerical, secondary_budget=None):
    """Compose approximation and integration error without relaxing rigid tests."""
    if secondary_budget is None:return numerical
    return max(numerical,min(.002,numerical+secondary_budget))


def secondary_sampling_tolerances(parts,understanding,accuracy_mode):
    """Component-owned particle sampling bounds, invariant to numerical grouping.

    Only explicitly secondary deformable semantic parts are eligible. Area
    centroids and the original surface span survive conforming subdivision.
    Primary cloth contacts and every rigid surface retain their original rules.
    """
    import numpy as np
    if mode(accuracy_mode)!='task_and_semantics':return {}
    eligible={link for item in understanding['parts'] if approximate_part(item)
              and item.get('task_primary') is False for link in item['links']}
    result={}
    for part in parts:
        if part['kind']!='cloth' or part['name'] not in eligible:continue
        points=np.asarray(part['V']);faces=np.asarray(part['F']);values=np.full(len(faces),-1.)
        components=part['cloth_rest'].get('components',[list(range(len(points)))])
        for indices in components:
            selected=np.isin(faces,indices).all(1);tri=points[faces[selected]]
            area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
            centre=np.einsum('n,nd->d',area,tri.mean(1))/area.sum()
            values[selected]=min(.002,.2*np.linalg.norm(points[indices]-centre,axis=1).max())
        if (values<=0).any():raise ValueError('Every secondary material triangle needs a finite owned sampling bound')
        result[part['name']]=values
    return result


def secondary_thickness_tolerances(parts,bindings,eligible,thickness):
    """Bound continuous-thickness diagnostics by a secondary particle model.

    The actual mid-surface still obeys the strict solid-penetration screen.
    This allowance cannot exceed half the declared material thickness, the
    smallest semantic-component budget or the measured particle sampling gap.
    Fully covered, primary, rigid and quantitative surfaces gain no allowance.
    """
    import numpy as np
    from .cloth_sampling import coverage
    understood={'parts':[{'representation':'deformable','task_primary':False,'links':list(eligible)}]}
    budgets=secondary_sampling_tolerances(parts,understood,'task_and_semantics');result={}
    for part in parts:
        name=part['name']
        if (name not in budgets or name not in thickness or part.get('backend','particle')!='particle'):continue
        faces=np.asarray(part['F']);bond=next((b for b in bindings if b['surface']==name),None)
        if bond:faces=faces[~np.isin(faces,bond['vertices']).any(1)]
        if not len(faces):continue
        half=thickness[name]/2;sampling=coverage(part['V'],faces,half)
        allowance=min(half,float(budgets[name].min()),sampling['maximum_unsampled_distance_bound_m'])
        if allowance>0:result[name]=float(allowance)
    return result
