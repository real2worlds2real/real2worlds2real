"""Source-only direction proposals before expensive flexible-body simulation.

Rotating a leaf sheet does not establish physical validity. Retain every
reviewed view, preserve the rigid assembly and material metric, then rebuild
and check actual contact clearance before adopting any proposal.
"""
import numpy as np
from scipy.spatial.transform import Rotation


from calibrated_scene_reconstruction.core.environment import setting

def rotate_link(vertices, parts, link, pivot, axis, degrees):
    vertices=np.asarray(vertices,float);pivot=np.asarray(pivot,float);axis=np.asarray(axis,float)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or pivot.shape!=(3,) or axis.shape!=(3,)
            or not np.isfinite(vertices).all() or not np.isfinite([pivot,axis]).all()
            or not np.isfinite(degrees) or not np.isclose(np.linalg.norm(axis),1.)):
        raise ValueError('Finite geometry, pivot, angle and unit rotation axis required')
    selected=[p for p in parts if p['name']==link]
    if len(selected)!=1 or selected[0]['kind']!='cloth':
        raise ValueError('Direction proposal requires one named flexible sheet; rigid mechanism frames are not rotated')
    part=selected[0];start,count=part['vertex_start'],part['vertex_count']
    if start<0 or count<=0 or start+count>len(vertices):raise ValueError('Invalid sheet vertex range')
    result=vertices.copy();rotation=Rotation.from_rotvec(axis*np.radians(degrees)).as_matrix()
    result[start:start+count]=(result[start:start+count]-pivot)@rotation.T+pivot
    return result


def evaluate(vertices,faces,pose,observations,parts,link,pivot,axis,angles,rasterizer=None):
    from .evidence import placed
    from .part_evidence import score
    sources=[p for p in observations[0].get('_source_parts',[]) if p['link']==link]
    if not sources:raise ValueError('Independent source sheet labels are required for direction review')
    angles=np.asarray(angles,float)
    if angles.ndim!=1 or not 1<=len(angles)<=48 or not np.isfinite(angles).all():
        raise ValueError('One to 48 finite direction hypotheses required')
    cameras={**observations[0],'_source_parts':sources};rows=[]
    for angle in angles:
        proposal=rotate_link(vertices,parts,link,pivot,axis,angle)
        metrics=score(placed(proposal,faces,pose),pose,cameras,parts,rasterizer)
        rows.append({'angle_deg':float(angle),'source_parts':metrics})
    return {'rows':rows,'pivot':np.asarray(pivot).tolist(),'axis':np.asarray(axis).tolist(),
            'proposal_only':True,'physical_validity_checked':False,'task_approved':False,
            'scope':'All reviewed source views; rigid geometry and intrinsic dimensions unchanged. Direction proposals require fresh finite-thickness, source-pose and native checks.'}


def run(session,obj,link,candidate,output,*,pivot,axis,angles):
    """Expose source direction comparison to the agent without adopting a pose."""
    import json
    from pathlib import Path
    from . import evidence
    from .material_reference import mesh_parts
    from .part_evidence import sheets
    import os
    from .projection import optional
    from .store import digest
    root=session.dir.resolve();candidate=Path(candidate).resolve();output=Path(output).resolve()
    if not candidate.is_relative_to(root) or not output.is_relative_to(root/'qa'):
        raise ValueError('Current-session candidate and QA output required')
    info=json.loads(candidate.read_text());source=(Path(info['path'])/'mesh.npz').resolve()
    if not source.is_relative_to(root):raise ValueError('Current-session geometry required')
    with np.load(source) as z:mesh={k:z[k].copy() for k in z.files}
    parts=mesh_parts(mesh);observations=evidence.load(session,obj,parts=True)
    rasterizer=None if setting('GEOMETRY_DEVICE')=='cpu' else optional()
    result=evaluate(mesh['V'],mesh['F'],info['pose'],observations,parts,link,pivot,axis,angles,rasterizer)
    best=max(result['rows'],key=lambda row:row['source_parts'][0]['iou_median'])
    mesh['V']=rotate_link(mesh['V'],parts,link,pivot,axis,best['angle_deg'])
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'mesh_proposal.npz',**mesh)
    sources=[p for p in observations[0]['_source_parts'] if p['link']==link]
    visual=sheets(session,evidence.placed(mesh['V'],mesh['F'],info['pose']),
        {**observations[0],'_source_parts':sources},parts,output/'source_comparison')
    result.update(ok=True,source_selected_angle_deg=best['angle_deg'],requires_agent_rebuild=True,
        mesh_proposal=str(output/'mesh_proposal.npz'),
        evidence_images=[p for item in visual.values() for p in item['sheets']],
        input_sha256={str(p):digest(p) for p in [candidate,source,Path(__file__)]})
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
