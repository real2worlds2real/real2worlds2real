"""Explicit, unapplied shader ablations on an existing scene's session layer."""
import math

from pxr import Gf, Sdf, UsdShade


def apply(stage, proposal):
    """Change only existing shader inputs; never geometry, bindings or physics.

    Connected inputs require an explicit disconnect declaration. Validate the
    entire proposal before changing anything, so a typo cannot partly apply it.
    The caller keeps the source USD immutable and fingerprints this declaration.
    """
    if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
        raise ValueError('Material proposals require the anonymous session layer')
    if not isinstance(proposal, dict) or not proposal.get('basis') or not proposal.get('shaders'):
        raise ValueError('Material proposal needs a basis and explicit shader inputs')
    pending = []
    vector_types = {'color3f': Gf.Vec3f, 'float3': Gf.Vec3f, 'float4': Gf.Vec4f,
                    'color3d': Gf.Vec3d, 'double3': Gf.Vec3d}
    for path, row in proposal['shaders'].items():
        shader = UsdShade.Shader(stage.GetPrimAtPath(path))
        if not shader or not row.get('inputs'):
            raise ValueError('Missing existing shader or inputs: '+path)
        disconnect = set(row.get('disconnect', []))
        if disconnect - set(row['inputs']):
            raise ValueError('Disconnect declaration must belong to an edited input')
        for name, value in row['inputs'].items():
            inp = shader.GetInput(name)
            if not inp:
                raise ValueError('Unknown existing shader input: '+path+'.'+name)
            if inp.HasConnectedSource() and name not in disconnect:
                raise ValueError('Connected shader input requires explicit disconnect: '+name)
            kind = str(inp.GetTypeName())
            if kind in vector_types:
                n = 4 if kind == 'float4' else 3
                if not isinstance(value, list) or len(value) != n:
                    raise ValueError('Shader vector has incorrect dimensions')
                value = vector_types[kind](*value)
            elif kind == 'bool':
                if not isinstance(value, bool):
                    raise ValueError('Boolean shader input requires bool')
            elif kind in {'float', 'double', 'int'}:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError('Numeric shader input requires a number')
                if kind == 'int' and not isinstance(value, int):
                    raise ValueError('Integer shader input requires int')
            else:
                raise ValueError('Only existing numeric shader inputs may be proposed: '+kind)
            values = list(value) if kind in vector_types else [value]
            if not all(math.isfinite(x) for x in values):
                raise ValueError('Shader input must be finite')
            pending.append((inp, value, name in disconnect))
    for inp, value, disconnect in pending:
        if disconnect:
            inp.DisconnectSource()
        inp.Set(value)
    return {'approved': False, 'scope': 'Material appearance proposal only',
            'basis': proposal['basis'], 'shaders': proposal['shaders']}
