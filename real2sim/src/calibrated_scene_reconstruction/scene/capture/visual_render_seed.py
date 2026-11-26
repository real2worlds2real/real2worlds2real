"""Broad root hypotheses from current labelled images and legal CAD only."""
import numpy as np


VISUAL_SEEDS = ('inferred', 'cad_home_hypothesis', 'midpoint_hypothesis',
                'quarter_home_hypothesis', 'three_quarter_home_hypothesis')


def visual_seed(state, home, name):
    """Independent CAD interpolation starts; never supplied execution telemetry."""
    fractions = dict(zip(VISUAL_SEEDS, (0., 1., .5, .25, .75)))
    if name not in fractions:
        raise ValueError('Unknown visual-only initialization hypothesis: '+name)
    fraction = fractions[name]
    return {key: (1-fraction)*value+fraction*home.get(key, 0.)
            for key, value in state.items()}


def root_hypothesis(renderer, state, observed, train_indices, initial):
    import torch
    make=renderer.kinematics.tensor
    q={name:make(value) for name,value in state.items()}
    local=renderer.kinematics.points(q,renderer.sample).detach().cpu().numpy()
    height=float(np.percentile(local[:,2],98)-renderer.floor)
    observed=np.asarray(observed,float)
    scale=max(.01,float(np.percentile(observed[:,2],98)/height))
    centre=np.median(observed[:,:2],axis=0)
    sample_centre=np.median(local[:,:2],axis=0)
    indices=[train_indices[i] for i in np.linspace(0,len(train_indices)-1,min(6,len(train_indices))).astype(int)]
    candidates=[]
    for yaw in np.radians(np.arange(0,360,30)):
        R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        for factor in (.75,1.,1.35):
            k=scale*factor
            candidates.append({'xy':(centre-k*sample_centre@R.T).tolist(),
                               'yaw':float(yaw),'scale':k})
    candidates.append({'xy':initial['base_xy_map'],'yaw':float(np.radians(initial['yaw_deg'])),
                       'scale':initial['map_units_per_metre']})
    with torch.no_grad():
        for candidate in candidates:
            vertices=renderer.world(q,make(candidate['xy']),make(candidate['yaw']),
                                    make(np.log(candidate['scale'])),renderer.mesh)
            prediction=renderer.render(vertices,indices)
            actual=renderer.mask[indices];valid=renderer.valid[indices]
            intersection=(prediction*actual*valid).sum((1,2))
            union=((prediction+actual-prediction*actual)*valid).sum((1,2)).clamp(min=1)
            candidate['training_iou']=float((intersection/union).median())
    candidates.sort(key=lambda c:c['training_iou'],reverse=True)
    return {**candidates[0],'hypotheses':len(candidates),'height_scale_hypothesis':scale,
            'source':'full-yaw/current-mask root search; CAD pose remains an unmeasured hypothesis',
            'top_candidates':candidates[:6]}


def resume_hypotheses(directory, previous, data, model):
    """Resume independent lineages only when capture/CAD observations are identical."""
    import json
    from pathlib import Path
    directory,previous=Path(directory).resolve(),Path(previous).resolve()
    # Prepared packages live at SESSION/splat/visual_render/round_N/input.
    capture=directory.parents[3]
    if not previous.is_relative_to(capture):
        raise ValueError('Cannot resume CAD calibration from a different capture session')
    old_input=previous.parent/'input'
    old=dict(np.load(old_input/'observations.npz'))
    if set(old)!=set(data) or any(not np.array_equal(old[key],data[key]) for key in data):
        raise ValueError('Resume requires identical current-capture observations')
    from .robot_visual import digest
    def metadata(path):
        return json.loads(path.read_text()) if path.exists() else {}
    current_meta=metadata(directory/'manifest.json');old_meta=metadata(old_input/'manifest.json')
    mode=current_meta.get('calibration_mode','visual_only')
    if mode!=old_meta.get('calibration_mode','visual_only'):
        raise ValueError('Resume cannot mix supplied-joint and joint-free calibration modes')
    joint_prior=mode=='joint_prior'
    if joint_prior:
        prior_path=directory/'joint_prior.json';old_prior=old_input/'joint_prior.json'
        if not prior_path.exists() or not old_prior.exists() or digest(prior_path)!=digest(old_prior):
            raise ValueError('Resume requires identical commanded-joint uncertainty assumptions')
    seeds=current_meta.get('seeds',('inferred','cad_home_hypothesis','midpoint_hypothesis'))
    result={}
    for seed in seeds:
        path=previous/(seed+'.json');candidate=json.loads(path.read_text())
        if (candidate.get('supplied_joint_config_read') is not joint_prior or any(
                candidate.get(key) is not False for key in ('board_input_read','measured_task_or_support_size_read'))):
            raise ValueError('Resume input provenance differs: supplied joints, board or task truth')
        if candidate.get('seed',seed)!=seed:
            raise ValueError('Resume must preserve each independent hypothesis identity')
        hashes=candidate['render_fit']['input_sha256']
        if joint_prior and hashes.get(str(old_prior))!=digest(old_prior):
            raise ValueError('Previous worker is not bound to this commanded-joint prior')
        for dependency in model.dependencies:
            if hashes.get(str(dependency))!=digest(dependency):
                raise ValueError('Resume CAD differs from the previous legal asset')
        result[seed]={'candidate':candidate,'path':str(path),'sha256':digest(path)}
    return result
