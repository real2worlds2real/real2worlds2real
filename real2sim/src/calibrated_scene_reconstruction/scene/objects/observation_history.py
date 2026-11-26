"""Preserve source labels before acquiring a new object-observation revision."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def color_inputs(session, cameras, frames):
    """Bind labels to actual RGB bytes and reject legacy HDR camera-image reuse."""
    from calibrated_scene_reconstruction.core.paths import ROOT
    from calibrated_scene_reconstruction.core.media import color_profile
    from calibrated_scene_reconstruction.scene.capture.supplementary import load
    from calibrated_scene_reconstruction.scene.capture.exposure import same_profile
    plan = load(session) if hasattr(session, 'config_dir') else None
    expected = None
    if plan:
        primary = next(item for item in plan['assets'] if item['id'] == plan['primary_scene'])
        if primary['kind'] == 'video':
            expected = color_profile(ROOT/primary['path'])
    source = Path(str(cameras['source']))
    if not source.is_absolute():
        source = ROOT/source
    color_path = source/'capture/color.json'
    if expected is not None and expected['mode'] == 'hdr_base_to_srgb':
        rectification = source/'splat/rectification.json'
        if (not color_path.exists() or not same_profile(expected, json.loads(color_path.read_text()))
                or not rectification.exists()
                or not same_profile(expected, json.loads(rectification.read_text()).get('color_profiles', {}).get('0'))):
            raise ValueError('Primary HDR camera images use legacy color; rebuild source frames and affected labels before fitting')
    if not color_path.exists():
        return None
    pixels = {}
    for index in frames:
        image = source/'splat/data/images'/str(cameras['names'][index])
        pixels[str(image)] = hashlib.sha256(image.read_bytes()).hexdigest()
    return {'profile_sha256': hashlib.sha256(color_path.read_bytes()).hexdigest(), 'images': pixels}


def check_color(data, expected):
    if expected is not None and ('source_color_inputs' not in data
            or json.loads(str(data['source_color_inputs'])) != expected):
        raise ValueError('Object observation source pixels/color changed; preserve and refresh labels')


def object_request(session, obj):
    graph = getattr(session, 'scene_graph', {})
    return graph.get('dynamic_objects', {}).get(obj, {}).get('fit') or {}


def semantic_policy(request):
    """Acquisition settings only; changing model shape does not relabel RGB."""
    anchors = request.get('part_anchors') or {}
    exclude = request.get('exclude_prompts', [])
    crop = request.get('crop_min')
    if (not isinstance(anchors, dict) or any(not isinstance(name, str) or not name.strip()
            or not isinstance(side, str) or side not in {'top','bottom','left','right','center'} for name, side in anchors.items())
            or not isinstance(exclude, list) or any(not isinstance(p, str) or not p.strip() for p in exclude)
            or (crop is not None and (type(crop) is not int or crop < 32))):
        raise ValueError('Invalid semantic part anchors, exclusion prompts or observation crop policy')
    result = {'part_anchors': anchors, 'exclude_prompts': exclude, 'crop_min': crop}
    confidence = request.get('instance_confidence_min')
    if confidence is not None:
        if type(confidence) not in (int, float) or not np.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError('Instance confidence minimum must be finite within0..1')
        result['instance_confidence_min'] = confidence
    collections = request.get('part_instances', {})
    parts = request.get('prompt', '').split('|')
    optional = request.get('optional_part_queries', [])
    if (not isinstance(optional, list)
            or any(not isinstance(name, str) or name not in parts[1:] for name in optional)
            or len(set(optional)) != len(optional)):
        raise ValueError('Optional queries must name distinct secondary parts; the primary assembly remains required')
    if optional:
        result['optional_part_queries'] = optional
    if (not isinstance(collections, dict) or any(
            name not in parts[1:] or not isinstance(spec, dict)
            or set(spec) != {'selection', 'gap_pixels'}
            or spec['selection'] != 'touching_primary'
            or type(spec['gap_pixels']) is not int or not 0 <= spec['gap_pixels'] <= 8
            for name, spec in collections.items())):
        raise ValueError('Part collections require a later query, touching_primary and integer gap0..8')
    context = request.get('collection_context')
    if context is not None:
        if (not isinstance(context,dict) or set(context)!={'object','padding_fraction','min_overlap_fraction'}
                or not isinstance(context['object'],str) or not context['object'].isidentifier()
                or any(type(context[k]) not in (int,float) for k in ['padding_fraction','min_overlap_fraction'])
                or not np.isfinite([context['padding_fraction'],context['min_overlap_fraction']]).all()
                or not 0<=context['padding_fraction']<=1 or not 0<context['min_overlap_fraction']<=1
                or collections or anchors or exclude):
            raise ValueError('Context collection requires one independent object, bounded association and distinct acquisition semantics')
        result['collection_context'] = context
    # Preserve byte-compatible policies for existing single-instance captures.
    if collections:
        result['part_instances'] = collections
    return result


def check_policy(data, request):
    expected = semantic_policy(request)
    if request.get('prompt') is not None:
        if 'prompt' not in data or str(data['prompt']) != request['prompt']:
            raise ValueError('Object prompt changed; preserve and refresh object observations explicitly')
    if 'semantic_query_policy' in data:
        if json.loads(str(data['semantic_query_policy'])) != expected:
            raise ValueError('Semantic query policy changed; preserve and refresh object observations explicitly')
    elif any(expected.values()):
        raise ValueError('Legacy object observations have no semantic query policy; refresh or bind an explicitly reviewed acquisition')


def bind_reviewed_policy(cache, request, basis):
    """Migrate reviewed acquisition metadata, preserving every source mask.

    This is an explicit agent review, not evidence inferred from model error or
    a method for automatically accepting old labels under different queries.
    """
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError('Explain why the reviewed labels used this exact query policy')
    cache = Path(cache)
    with np.load(cache, allow_pickle=True) as data:
        payload = {name: data[name] for name in data.files}
    expected = semantic_policy(request)
    if request.get('prompt') is not None and str(payload.get('prompt', '')) != request['prompt']:
        raise ValueError('Object prompt changed; do not relabel source masks through a metadata migration')
    if 'semantic_query_policy' in payload:
        check_policy(payload, request)
        raise ValueError('Acquisition policy is already bound; do not overwrite a reviewed revision')
    before = hashlib.sha256(cache.read_bytes()).hexdigest()
    previous = archive(cache, {'semantic_query_policy_review': basis})
    payload['semantic_query_policy'] = json.dumps(expected, sort_keys=True)
    payload['semantic_query_policy_review'] = json.dumps({'basis': basis, 'previous_sha256': before,
                                                         'previous_revision': str(previous)})
    temporary = cache.with_suffix('.policy.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(cache)
    return previous


def archive(cache, reason):
    cache = Path(cache)
    if not cache.exists():
        return None
    root = cache.parent/'history'/cache.stem
    number = max([int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()], default=0)+1
    destination = root/f'round_{number}'
    destination.mkdir(parents=True, exist_ok=False)
    sources = [p for p in cache.parent.glob(cache.stem+'.*') if p.is_file()]
    hashes = {}
    for source in sources:
        shutil.copy2(source, destination/source.name)
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (destination/'revision.json').write_text(json.dumps({'reason': reason, 'source_sha256': hashes}, indent=2)+'\n')
    return destination
