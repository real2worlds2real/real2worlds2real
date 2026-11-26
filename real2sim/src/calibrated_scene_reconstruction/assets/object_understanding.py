"""Source-backed asset structure and representation contracts."""
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def static_support(session,obj):
    return obj in getattr(session,'scene_graph',{}).get('furniture',{})


def _files(session, files, *, qa=False):
    if not isinstance(files, dict) or not files:
        raise ValueError('Understanding evidence needs hashed source or QA files')
    root = (session.dir / 'qa' if qa else session.dir).resolve()
    for name, sha in files.items():
        path = (session.dir / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != sha:
            raise ValueError('Understanding evidence missing, changed or outside session')


def declaration(session, obj, *, plan=None):
    # Explicit candidate snapshots must be validated themselves, not replaced
    # by a file inferred from their parent directory. Source-file hashes and
    # every semantic/physical obligation below are identical for either route.
    if plan is None:
        if not hasattr(session, 'config_dir'):
            raise ValueError('Object understanding requires a scene session with its configuration directory')
        path = Path(session.config_dir) / 'agent/object_understanding.json'
        if not path.is_file():
            raise ValueError('Agent must declare agent/object_understanding.json before modeling')
        plan = json.loads(path.read_text())
    if not isinstance(plan, dict):raise ValueError('Complete object-understanding plan required')
    item = plan.get('objects', {}).get(obj)
    if plan.get('version') != 1 or not isinstance(item, dict):
        raise ValueError('Every reconstructed object needs source-grounded understanding')
    if any(not _text(item.get(k)) for k in ('identity', 'source_coverage', 'uncertainties')):
        raise ValueError('Declare object identity, source coverage and uncertainties')
    _files(session, item.get('source_evidence'))
    parts = item.get('parts')
    if not isinstance(parts, list) or not parts:
        raise ValueError('Inventory all semantic parts, including task-secondary parts')
    ids = [p.get('id') for p in parts if isinstance(p, dict)]
    if len(ids) != len(parts) or any(not _text(x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Semantic parts need distinct ids')
    tests = []
    structural=item.get('structural_priors',[])
    if (not isinstance(structural,list) or any(not isinstance(p,dict)
            or any(not _text(p.get(k)) for k in ('id','description','source_basis'))
            or p.get('kind') not in {'part_inventory','layout','connections','support','material_behavior'} for p in structural)
            or len({p['id'] for p in structural})!=len(structural)):
        raise ValueError('Structural priors need distinct source-grounded checks')
    if static_support(session,obj) and not {'part_inventory','layout','connections','support'}<={p['kind'] for p in structural}:
        raise ValueError('Furniture needs part inventory, layout/symmetry, connections and support checks')
    for part in parts:
        from calibrated_scene_reconstruction.assets.response_policy import mode,approximate_part
        mode(part.get('response_accuracy'))
        for key in ('structure', 'connection', 'material_hypothesis', 'material_alternatives',
                    'response_basis', 'representation_reason', 'geometry_feature'):
            if not _text(part.get(key)):
                raise ValueError('Each part needs structure, material, response and representation reasoning')
        if type(part.get('task_primary')) is not bool:
            raise ValueError('Explicitly distinguish primary and secondary parts without dropping either')
        if part.get('expected_response') not in {'rigid', 'flexible', 'articulated', 'uncertain'}:
            raise ValueError('Declare expected mechanical response independently of chosen representation')
        if part.get('representation') not in {'rigid', 'deformable', 'articulated'}:
            raise ValueError('Choose a physical representation for each semantic part')
        if part.get('basis') not in {'observed', 'inferred'}:
            raise ValueError('Separate observed structure from inferred construction')
        if part.get('connection_type') not in {'integrated','bonded','joint','loose','support','uncertain'}:
            raise ValueError('Declare physical connection type independently of the task')
        if not isinstance(part.get('links'), list) or not part['links'] or any(not _text(x) for x in part['links']):
            raise ValueError('Map every semantic part to actual modeled physical links')
        _files(session, part.get('source_evidence'))
        obligations = part.get('physics_tests', [])
        if not isinstance(obligations,list):raise ValueError('Physical test inventory must be a list')
        purposes = {x.get('purpose') for x in obligations}
        required = set() if static_support(session,obj) or approximate_part(part) else {'passive_state', 'contact_response', 'material_response'}
        if part['expected_response'] == 'articulated' or part['representation'] == 'articulated':
            required.add('joint_response')
        if not required <= purposes:
            raise ValueError('Every part needs non-task passive, contact and material response tests')
        for test in obligations:
            if not _text(test.get('id')) or not _text(test.get('scenario')):
                raise ValueError('Physical tests need explicit scenarios and unique ids')
            metrics = test.get('metrics')
            if not isinstance(metrics, list) or not metrics:
                raise ValueError('Physical checks need measured quantities and prior bounds')
            for metric in metrics:
                bounds = metric.get('bounds')
                if (any(not _text(metric.get(k)) for k in ('name', 'unit', 'basis'))
                        or not isinstance(bounds, list) or len(bounds) != 2
                        or any(type(x) not in (int, float) for x in bounds)
                        or not np.isfinite(bounds).all() or bounds[0] > bounds[1]):
                    raise ValueError('Physical metrics need finite ordered prior bounds, units and basis')
            if len({m['name'] for m in metrics}) != len(metrics):
                raise ValueError('Duplicate physical metric')
            tests.append(test['id'])
    if len(set(tests)) != len(tests):
        raise ValueError('Physical test ids must be unique across parts')
    for part in parts:
        if part['representation'] == 'rigid' and part['expected_response'] in {'flexible', 'uncertain'}:
            approximation = part.get('rigid_approximation', {})
            if not _text(approximation.get('basis')) or not _text(approximation.get('error_metric')):
                raise ValueError('Rigid approximation needs an independently tested physical error bound')
            metrics = [m for t in part['physics_tests'] if t['purpose']=='material_response'
                       for m in t['metrics']]
            if approximation['error_metric'] not in {m['name'] for m in metrics}:
                raise ValueError('Rigid approximation error must be measured by a material response test')
    return item


def validate_build(session, obj, build, *, preflight=False, understanding_plan=None):
    item = declaration(session, obj, plan=understanding_plan)
    parts = build.get('parts', [])
    # Legacy static-support builders report visual construction ranges, not
    # dynamic link kinds. The native support author has one static model owner.
    # Normalize only that factual ownership, without inventing a semantic pass
    # or extending the fallback to dynamic/flexible/articulated geometry.
    legacy_static = (static_support(session, obj) and parts
                     and all('kind' not in part for part in parts))
    if legacy_static:
        required = {'name', 'vertex_start', 'vertex_count', 'triangle_count'}
        allowed = required | {'face_start', 'material'}
        if (build.get('links') or build.get('joints') or build.get('surface_bindings')
                or any(not required <= set(part) or set(part) - allowed or not _text(part['name'])
                       or type(part['vertex_start']) is not int or part['vertex_start'] < 0
                       or type(part['vertex_count']) is not int or part['vertex_count'] <= 0
                       or type(part['triangle_count']) is not int or part['triangle_count'] <= 0
                       for part in parts)):
            raise ValueError('Legacy static support needs genuine visual ranges and no physical links or bindings')
        links = {'model': 'rigid'}
    else:
        if any('kind' not in part or 'name' not in part for part in parts):
            raise ValueError('Explicit physical link kinds required outside legacy static support')
        links = {p['name']: p['kind'] for p in parts}
    if not links:
        raise ValueError('Understanding requires explicit physical link ownership')
    if 'links' in build and build['links'] is None:
        # Blender's original rigid-only path unions source meshes into one
        # physical owner. Its mesh object names are visual construction names,
        # whereas metric_mesh.asset and the stage author use owner 'model'.
        if set(links.values())!={'rigid'} or build.get('joints') or build.get('surface_bindings'):
            raise ValueError('Implicit model ownership supports only a single rigid assembly')
        links={'model':'rigid'}
    covered = set()
    for part in item['parts']:
        covered.update(part['links'])
        if not set(part['links']) <= links.keys():
            raise ValueError('Semantic part maps to missing physical link')
        kinds = {links[x] for x in part['links']}
        flexible = bool(kinds & {'cloth', 'deformable', 'rope'})
        if (part['representation'] == 'deformable') != flexible:
            raise ValueError('Declared flexibility differs from actual modeled link kinds')
        if not preflight and part['representation'] == 'articulated' and not build.get('joints'):
            raise ValueError('Articulated representation has no modeled connections')
        if not preflight and part['connection_type']=='bonded' and part['representation']=='deformable':
            rooted={b['surface'] for b in build.get('surface_bindings',[])}
            if not set(part['links'])<=rooted:
                raise ValueError('Rooted flexible parts need actual exported physical bindings')
    if covered != links.keys():
        raise ValueError('Every actual physical link must appear in semantic inventory')
    return item
