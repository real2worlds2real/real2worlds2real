"""A bounded persistent Blender subprocess, geometry validation and content-addressed mesh cache."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from calibrated_scene_reconstruction.assets.metric_mesh import asset_geometry, link_geometry

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.objects.blender.contract import contract, geometry, parameters, write_json


class Blender:
    def __init__(self, model, output, timeout=180, *, max_extent_m=2., source_files=(),proposal_fixed=None):
        self.model, self.output, self.timeout = Path(model).resolve(), Path(output).resolve(), timeout
        self.spec = contract(model)
        self.proposal_fixed={} if proposal_fixed is None else dict(proposal_fixed)
        for name,value in self.proposal_fixed.items():
            original=self.spec['fixed'].get(name)
            if (name not in self.spec['fixed'] or isinstance(value,bool) or isinstance(original,bool)
                    or type(value) not in (int,float) or type(original) not in (int,float)
                    or not np.isfinite(value) or isinstance(original,int) and not isinstance(value,int)):
                raise ValueError('Proposal overrides must name finite numeric FIXED values with compatible types')
        self.max_extent_m = max_extent_m
        self.source_inputs = {**self.spec.get('dependencies', {}),
                              **{str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for path in source_files}}
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = (self.output / 'blender.log').open('a')
        self.process = None

    def _current_sources(self):
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
               for path, expected in self.source_inputs.items()):
            raise ValueError('Declared Blender source asset changed during this run')
        if contract(self.model) != self.spec:
            raise ValueError('Model source or imported geometry helper changed during this run')

    def __enter__(self):
        return self

    def evaluate(self, params=None, artifacts=False):
        self._current_sources()
        if artifacts and self.proposal_fixed:
            raise ValueError('Proposal-resolution geometry cannot produce delivery artifacts; rebuild the full model')
        values = parameters(self.spec, params)
        values.update(self.proposal_fixed)
        harness_hash = hashlib.sha256(Path(__file__).with_name('blender_harness.py').read_bytes()).hexdigest()
        root_hash = hashlib.sha256(Path(__file__).with_name('root_geometry.py').read_bytes()).hexdigest()
        validation_hash = hashlib.sha256(Path(__file__).resolve().parents[3].joinpath('assets/metric_mesh.py').read_bytes()).hexdigest()
        cache_input = [self.spec['sha256'], harness_hash, root_hash, validation_hash, values, artifacts]
        cache_input.append({'subsurface_optics': hashlib.sha256(
            Path(__file__).resolve().parents[3].joinpath('assets/subsurface_optics.py').read_bytes()).hexdigest()})
        cache_input.append({'mesh_numeric': hashlib.sha256(
            Path(__file__).resolve().parents[3].joinpath('assets/mesh_numeric.py').read_bytes()).hexdigest(),
            'mesh_normals':hashlib.sha256(
            Path(__file__).resolve().parents[3].joinpath('assets/mesh_normals.py').read_bytes()).hexdigest()})
        cache_input.append({'surface_bindings': hashlib.sha256(
            Path(__file__).resolve().parents[3].joinpath('assets/surface_bindings.py').read_bytes()).hexdigest(),
            'bond_footprint':hashlib.sha256(
            Path(__file__).resolve().parents[3].joinpath('assets/bond_footprint.py').read_bytes()).hexdigest(),
            'attachment':hashlib.sha256(Path(__file__).resolve().parents[3].joinpath('sim/surface_attachment.py').read_bytes()).hexdigest()})
        cache_input.append({'cloth_rest_validation': hashlib.sha256(
            Path(__file__).resolve().parents[3].joinpath('assets/cloth_rest.py').read_bytes()).hexdigest()})
        if any(link.get('backend')=='volume_fem' for link in (self.spec.get('links') or {}).values()):
            cache_input.append({'volume_initial_validation':hashlib.sha256(
                Path(__file__).resolve().parents[3].joinpath('assets/volume_initial_state.py').read_bytes()).hexdigest(),
                'tetra_boundary':hashlib.sha256(Path(__file__).resolve().parents[3].joinpath('assets/tetra_boundary.py').read_bytes()).hexdigest()})
        if any(link['kind']=='cloth' for link in (self.spec.get('links') or {}).values()):
            cache_input.append({'surface_mass':hashlib.sha256(
                Path(__file__).resolve().parents[3].joinpath('assets/surface_mass.py').read_bytes()).hexdigest()})
            cache_input.append({'surface_conditioning':hashlib.sha256(
                Path(__file__).resolve().parents[3].joinpath('assets/surface_conditioning.py').read_bytes()).hexdigest(),
                'surface_remesh':hashlib.sha256(Path(__file__).with_name('surface_remesh.py').read_bytes()).hexdigest()})
            cache_input.append({'particle_lattice_validation':hashlib.sha256(
                Path(__file__).resolve().parents[3].joinpath('assets/particle_lattice.py').read_bytes()).hexdigest()})
            cache_input.append({'material_outline_validation': hashlib.sha256(
                Path(__file__).with_name('material_reference.py').read_bytes()).hexdigest(),
                'cloth_outline':hashlib.sha256(Path(__file__).resolve().parents[3].joinpath('assets/cloth_outline.py').read_bytes()).hexdigest()})
        if self.max_extent_m != 2.:
            cache_input.append({'max_extent_m': self.max_extent_m})
        if self.source_inputs:
            cache_input.append({'source_inputs': self.source_inputs})
        key = hashlib.sha256(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
        dest = self.output / key
        if not (dest / 'report.json').exists():
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'response.json').unlink(missing_ok=True)
            if self.process is None:
                self.process = subprocess.Popen([str(ROOT / 'tools/run_blender.sh'), '--python',
                                                str(Path(__file__).with_name('blender_harness.py'))],
                                               stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                                               text=True, start_new_session=True, env={**os.environ, 'MEM': '40G'})
            self.process.stdin.write(json.dumps({'model': str(self.model), 'output': str(dest),
                                                'params': values, 'artifacts': artifacts,
                                                'links': self.spec.get('links'), 'joints': self.spec.get('joints', [])}) + '\n')
            self.process.stdin.flush()
            end = time.monotonic() + self.timeout
            while not (dest / 'response.json').exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f'Blender exited {self.process.returncode}; see {self.output / "blender.log"}')
                if time.monotonic() > end:
                    self.close()
                    raise TimeoutError(f'Blender exceeded {self.timeout} seconds')
                time.sleep(0.05)
            result = json.loads((dest / 'response.json').read_text())
            if not result['ok']:
                raise ValueError(result['error'])
            if self.spec.get('links'):
                from calibrated_scene_reconstruction.assets.mechanisms import joints
                joints(result.get('joints', []), self.spec['links'])
            with np.load(dest / 'mesh.npz') as mesh:
                result['geometry'] = asset_geometry(mesh['V'], mesh['F'], result['parts'] if self.spec.get('links') else None,
                                                    max_extent_m=self.max_extent_m)
                for part in result['parts'] if self.spec.get('links') else []:
                    a, n = part['vertex_start'], part['vertex_count']
                    b, count = part['face_start'], part['triangle_count']
                    part['geometry'] = link_geometry(mesh['V'][a:a+n], mesh['F'][b:b+count]-a, part['kind'])
                    rest = json.loads(str(mesh['cloth_rest_json'])) if 'cloth_rest_json' in mesh else {}
                    if part['name'] in rest:
                        from calibrated_scene_reconstruction.assets.cloth_rest import validate as validate_rest
                        material = validate_rest(rest[part['name']], mesh['V'][a:a+n], mesh['F'][b:b+count]-a)
                        from .material_reference import intrinsic_geometry
                        part['geometry'].update(intrinsic_geometry(rest[part['name']],mesh['F'][b:b+count]-a,material['rest_area_m2']))
                        if part.get('backend')=='surface_fem':
                            from calibrated_scene_reconstruction.assets.surface_mass import distribution
                            mass_model=distribution(rest[part['name']]['points'],mesh['F'][b:b+count]-a,1.)
                            if not mass_model['pass']:raise ValueError('Uniform native node masses cannot represent these separate sheet masses; split physical bodies or remesh by material area')
                            part['geometry']['native_mass_discretization']=mass_model
            op = result.get('opening')
            if op is not None:
                if set(op) != {'xy', 'rim_z', 'depth', 'radius'} or len(op['xy']) != 2:
                    raise ValueError('OPENING requires xy, rim_z, depth, radius')
                if not np.isfinite([*op['xy'], op['rim_z'], op['depth'], op['radius']]).all() or min(op['depth'], op['radius']) <= 0:
                    raise ValueError('OPENING must have finite positive radius and depth')
            result['params'] = values
            result['model_sha256'] = self.spec['sha256']
            self._current_sources()
            write_json(dest / 'report.json', result)
        result = json.loads((dest / 'report.json').read_text())
        if self.proposal_fixed:
            result['proposal_only']=True
            result['proposal_fixed']=dict(self.proposal_fixed)
        if self.spec.get('links'):
            from calibrated_scene_reconstruction.assets.optical_surfaces import annotate
            from calibrated_scene_reconstruction.assets.surface_regions import annotate as regions
            with np.load(dest/'mesh.npz') as data:
                annotate(data,result['parts']);regions(data,result['parts'])
        self._current_sources()
        return dest, result

    def close(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop": true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                # The worker is a separate process group owned by this object; terminate only this group's PID.
                import signal
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        if self.process and self.process.stdin:
            self.process.stdin.close()
        self.log.close()

    def __exit__(self, *_):
        self.close()
