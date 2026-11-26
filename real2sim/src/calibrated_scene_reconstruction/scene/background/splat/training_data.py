"""Isolated GS training datasets; never rewrite capture or frozen loss masks."""
import json
from pathlib import Path
import shutil

from calibrated_scene_reconstruction.scene.review.environment import digest


def prepare(session,settings):
    import pycolmap
    from .config import subset_model
    src=session/'splat/data';root=session/'splat/training_data';root.mkdir(parents=True,exist_ok=True)
    rec=pycolmap.Reconstruction(str(src/'sparse/0'));names=sorted(img.name for img in rec.images.values())
    cfg=settings['train'];skip=set();mask_src=src;mask_inputs={}
    if cfg.get('mask_source_session'):
        from calibrated_scene_reconstruction.core.paths import resolve
        from calibrated_scene_reconstruction.scene.session import Session
        donor=Session(resolve(cfg['mask_source_session']));mask_src=donor/'splat/data'
        cameras=[session/'parsing/posed_frames/cameras.npz',donor/'parsing/posed_frames/cameras.npz']
        if any(not p.is_file() for p in cameras) or digest(cameras[0])!=digest(cameras[1]):
            raise ValueError('Mask reuse requires the exact same calibrated capture')
        def signature(graph):
            return {'task':graph.get('task'), 'objects':{k:v.get('ids') for k,v in graph.get('dynamic_objects',{}).items()},
                    'robot_ids':(graph.get('photo_robot') or {}).get('ids')}
        if signature(session.config('scene_graph.json'))!=signature(donor.config('scene_graph.json')):
            raise ValueError('Mask reuse requires the same task instance identities')
        for p in [*cameras,session.config_dir/'scene_graph.json',donor.config_dir/'scene_graph.json']:
            mask_inputs[str(p.resolve())]=digest(p)
    if cfg.get('skip_videos'):
        capture=json.loads((session/'capture/keyframes.json').read_text())
        videos=[capture['video']]+[v for v in settings['videos'] if v!=capture['video']]
        skip={str(i) for i,v in enumerate(videos) if v in cfg['skip_videos']}
    names=[n for n in names if n.split('_')[0] not in skip]
    if len(names)<2:raise ValueError('At least two registered training cameras required')
    if any(Path(n).name!=n or Path(n).suffix.lower()!='.jpg' for n in names):raise ValueError('Registered JPG basenames required')
    planned=[];legacy=not (mask_src/'loss_masks.json').exists()
    for name in names:
        source=src/'images'/name
        if not source.is_file():raise ValueError('Registered training image missing: '+name)
        planned.append((source,Path('images')/name))
        donor_image=mask_src/'images'/name
        if mask_src!=src and (not donor_image.is_file() or digest(donor_image)!=digest(source)):
            raise ValueError('Mask donor source pixels differ: '+name)
        sam=donor_image.with_name(donor_image.stem+'_sam.png')
        if not sam.is_file() and legacy:sam=donor_image.with_name(donor_image.stem+'_mask.png')
        if cfg.get('masked') and not sam.is_file():raise ValueError('Background-only training requires a task exclusion mask for every registered frame: '+name)
        if sam.is_file():planned.append((sam,Path('images')/(source.stem+'_sam.png')))
        for suffix in ('_zone.png','_valid.png'):
            mask=source.with_name(source.stem+suffix)
            if mask.is_file():planned.append((mask,Path('images')/mask.name))
    numbers=[int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
    out=root/f'round_{max(numbers,default=0)+1}';out.mkdir();(out/'images').mkdir()
    inputs=dict(mask_inputs);snapshot={}
    for source,relative in planned:
        before=digest(source);target=out/relative;shutil.copy2(source,target)
        if digest(target)!=before or digest(source)!=before:raise ValueError('Training observation changed during snapshot')
        inputs[str(source.resolve())]=before;snapshot[str(target.resolve())]=before
    if skip:
        subset_model(src/'sparse/0',out/'sparse/0',lambda n:n in set(names))
    else:shutil.copytree(src/'sparse',out/'sparse',symlinks=False)
    for p in (out/'sparse').rglob('*'):
        if p.is_file():snapshot[str(p.resolve())]=digest(p)
    proof={'version':1,'source_inputs':inputs,'snapshot_sha256':snapshot,'registered_frames':names,
           'settings':settings,'source_data':str(src.resolve()),
           'scope':'Exact registered capture observations and poses copied before training; no new frame selection, calibration or task geometry. Loss masks are written only in this private dataset.'}
    (out/'snapshot.json').write_text(json.dumps(proof,indent=2)+'\n');return out


def validate(data):
    path=Path(data)/'snapshot.json';proof=json.loads(path.read_text())
    for p,expected in proof['snapshot_sha256'].items():
        if digest(p)!=expected:raise ValueError('Training snapshot changed: '+p)
    return {'path':str(path.resolve()),'sha256':digest(path),'frames':len(proof['registered_frames'])}


def seal_masks(data):
    path=Path(data)/'snapshot.json';proof=json.loads(path.read_text())
    paths=[Path(data)/'loss_masks.json',*sorted((Path(data)/'images').glob('*_mask.png'))]
    proof['snapshot_sha256'].update({str(p.resolve()):digest(p) for p in paths})
    path.write_text(json.dumps(proof,indent=2)+'\n');return validate(data)
