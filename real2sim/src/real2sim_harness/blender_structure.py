"""Capture authored structure and visible geometry without changing the source."""
from __future__ import annotations

import hashlib
import json

import bpy
import numpy as np
from blender_surface import is_asset_object


import sys
from pathlib import Path

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_key, asset_value


def snapshot_structure():
    joints = [obj for obj in bpy.context.scene.objects if obj.get('joint_type')]
    rows = []
    for joint in joints:
        parent = joint.parent
        while parent is not None and not parent.get('joint_type'):
            parent = parent.parent
        rows.append({'name': joint.name, 'joint_type': str(joint['joint_type']).lower(),
                     'parent_joint': parent.name if parent else None})
    loops = json.loads(asset_value(bpy.context.scene, 'real2sim_loop_constraints', '[]'))
    motion = json.loads(asset_value(bpy.context.scene, 'real2sim_hybrid_motion', '[]'))
    attachments = []
    for obj in bpy.context.scene.objects:
        if asset_value(obj, 'real2sim_attachments'):
            attachments.extend({'control_object': obj.name, 'group': row['group'], 'body': row['body']}
                               for row in json.loads(obj[asset_key(obj, 'real2sim_attachments')]))
        elif any(modifier.type == 'CLOTH' for modifier in obj.modifiers):
            for modifier in obj.modifiers:
                if modifier.type == 'HOOK' and modifier.object and modifier.vertex_group:
                    attachments.append({'control_object': obj.name, 'group': modifier.vertex_group,
                                        'body': modifier.object.name})
    vertices = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or not is_asset_object(obj):
            continue
        coordinates = np.empty(len(obj.data.vertices)*3, dtype=np.float64)
        obj.data.vertices.foreach_get('co', coordinates)
        matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        vertices.append(coordinates.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3])
    if not vertices:
        raise ValueError('A functional asset needs visible mesh geometry')
    points = np.concatenate(vertices)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite cousin geometry')
    size = np.ptp(points, axis=0)
    points -= (points.min(axis=0)+points.max(axis=0))/2
    canonical = np.unique(np.round(points, 6), axis=0)
    # Integer micrometres avoid negative-zero and float byte-order ambiguities.
    canonical = np.rint(canonical*1e6).astype('<i8')
    has_cloth = any(any(modifier.type == 'CLOTH' for modifier in obj.modifiers) for obj in bpy.context.scene.objects)
    inferred_kind = 'hybrid' if joints and has_cloth else 'articulated' if joints else 'deformable' if has_cloth else 'rigid'
    asset_contract = {'kind': asset_value(bpy.context.scene, 'real2sim_asset_kind', inferred_kind),
                      'front_axis': asset_value(bpy.context.scene, 'real2sim_front_axis', '+Y'),
                      'up_axis': asset_value(bpy.context.scene, 'real2sim_up_axis', '+Z'),
                      'units': asset_value(bpy.context.scene, 'real2sim_units', 'm'), 'required_parts': []}
    return {'schema_version': 1, 'asset_contract': asset_contract,
            'joints': sorted(rows, key=lambda row: row['name']),
            'loop_constraints': [{key: row[key] for key in ('type', 'body_a', 'body_b')} for row in loops],
            'cloth_attachments': attachments, 'controlled_joints': sorted(row['joint'] for row in motion),
            'geometry_fingerprint': hashlib.sha256(canonical.tobytes()).hexdigest(),
            'visible_extent_m': size.tolist()}
