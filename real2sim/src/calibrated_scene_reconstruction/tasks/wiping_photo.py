"""External RGB-to-sim input. No Isaac runtime, particle state, or hardware API."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.tasks.wiping_rgb import (TableCamera, WipeCoveragePlanner, detect_blue_water,
                                     detect_table_water, projected_robot_mask, mask_to_grid)


def rigid_transform(value):
    T = np.asarray(value, float)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(T[:3, :3]) < .99):
        raise ValueError('Expected a rigid transform in metres')
    return T


def photo_camera(intrinsics, resolution, geometry, robot_spec=None, *, image_domain='raw'):
    """Resolve the pose at exposure, either explicit or from measured joint FK."""
    if geometry.get('units') != {'length': 'm', 'angle': 'rad'}:
        raise ValueError('Geometry must declare metres and radians')
    if not geometry.get('frame'):
        raise ValueError('Name the shared scene coordinate frame')
    table = np.asarray(geometry['table_box'], float)
    if table.shape != (2, 3) or not np.isfinite(table).all() or not (table[1] > table[0]).all():
        raise ValueError('Expected a positive axis-aligned table box in the declared frame')
    root = rigid_transform(geometry['T_frame_robot_root'])
    if 'T_frame_color' in geometry:
        pose = rigid_transform(geometry['T_frame_color'])
        source = 'Supplied camera optical pose at exposure'
    else:
        from calibrated_scene_reconstruction.robots.articulated_model import from_usd
        if robot_spec is None:
            raise ValueError('Robot spec required to resolve camera FK')
        model = from_usd(robot_spec['usd'], points_per_link=1)
        measured = geometry['joint_positions_rad']
        by_child = {j.child: j for j in model.joints}
        extrinsics = geometry.get('camera_extrinsics')
        if extrinsics is not None:
            link = extrinsics['parent_link']
            local = rigid_transform(extrinsics['T_parent_color'])
            source = 'Measured joints + supplied parent-to-optical extrinsics'
        else:
            mount = next(c for c in robot_spec['cameras'] if c.get('role') == 'head')
            link = mount['link']
            local = np.eye(4)
            w, x, y, z = mount['quat_wxyz']
            local[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix() @ np.diag([1., -1, -1])
            local[:3, 3] = mount.get('pos', [0, 0, 0])
            source = 'Measured joints + URDF-derived camera mount (hardware extrinsics unvalidated)'
        ancestors = []
        cursor = link
        while cursor != model.root:
            if cursor not in by_child:
                raise ValueError(f'Camera link {link} is outside the robot tree')
            joint = by_child[cursor]
            if joint.kind != 'fixed':
                ancestors.append(joint.name)
            cursor = joint.parent
        missing = set(ancestors) - set(measured)
        if missing:
            raise ValueError(f'Missing exposure-time camera-chain joints: {sorted(missing)}')
        transforms = model.transforms({name: measured[name] for name in ancestors})
        pose = root @ transforms[link] @ local
    selected = intrinsics[resolution]
    distortion = selected.get('dist_coeffs', intrinsics['dist_coeffs'])
    if image_domain not in ['raw', 'rectified']:
        raise ValueError('Declare raw or rectified image coordinates')
    calibrated_domain = selected.get('pixel_domain', intrinsics.get('pixel_domain', 'raw'))
    if calibrated_domain != image_domain:
        raise ValueError('Image domain does not match calibration; rectified images need their rectified K')
    if image_domain == 'rectified' and np.any(np.asarray(distortion) != 0):
        raise ValueError('Rectified calibration must have zero distortion')
    camera = TableCamera(selected['K'], distortion,
                         tuple(selected['image_size']), pose, table[1, 2], table[:, :2])
    return camera, root, source


def export_wet_region(path, grid, origin, cell, table_z, frame):
    """Metric footprint mesh to sublayer into the known sim scene, not a volume estimate."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = stage.DefinePrim('/ObservedWater', 'Xform')
    stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(stage, '/ObservedWater/Footprint')
    y, x = np.nonzero(grid)
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    xy = np.asarray(origin) + (np.c_[x, y][:, None] + corners[None]) * cell
    vertices = np.c_[xy.reshape(-1, 2), np.full(len(x) * 4, table_z + .0002)]
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(x)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(len(vertices)))))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    root.SetCustomDataByKey('meaning', 'RGB-derived tabletop footprint; display offset 0.2mm; thickness/volume unknown')
    root.SetCustomDataByKey('coordinateFrame', frame)
    material = UsdShade.Material.Define(stage, '/ObservedWater/Material')
    shader = UsdShade.Shader.Define(stage, '/ObservedWater/Material/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.08, .35, .48))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.12)
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(.65)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def localize_photo(image, camera, root, output, *, source, frame, brush_size=(.035, .010), history=None,
                   image_domain='raw', exclude_mask=None, margin=(.018, .025)):
    image, output = Path(image), Path(output)
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Cannot read RGB image: {image}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    coarse = detect_blue_water(rgb, camera)
    mask = detect_table_water(rgb, camera, history=history, exclude_mask=exclude_mask)
    grid, origin, cell = mask_to_grid(mask, camera)
    planner = WipeCoveragePlanner(brush_size, margin=margin)
    # This exports the current image's candidate paths. Persistent, per-action
    # closed-loop verification remains the demo/controller's responsibility.
    strokes, _ = planner.update(grid, origin, cell, fine=0 < int(mask.sum()) < 100)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [origin + (c[:, 0] + .5) * cell for c in contours]
    gy,gx = np.nonzero(grid)
    bounds = (np.array([origin + np.array([gx.min(),gy.min()])*cell,
                        origin + np.array([gx.max()+1,gy.max()+1])*cell]) if len(gx) else None)
    inverse = np.linalg.inv(root)
    polygons_base = [(np.c_[p, np.full(len(p), camera.table_z)] - root[:3, 3]) @ root[:3, :3] for p in polygons]
    yy, xx = np.nonzero(mask)
    pixels = np.c_[xx, yy]
    points, valid = camera.pixel_to_table(pixels)
    if len(points):
        projected, _ = camera.project(points[valid])
        pixel_error = float(np.max(np.linalg.norm(projected - pixels[valid], axis=1)))
    else:
        pixel_error = None
    merged_history = grid.copy()
    if history is not None:
        previous, previous_origin, previous_cell = history
        if previous.shape != grid.shape or not np.allclose(previous_origin, origin) or previous_cell != cell:
            raise ValueError('History and current camera must use the same metric table grid')
        merged_history |= previous
    output.mkdir(parents=True, exist_ok=False)
    overlay = rgb.copy()
    overlay[mask > 0] = np.array([0, 170, 245], np.uint8)
    for stroke in strokes:
        uv, ok = camera.project(np.c_[stroke, np.full(2, camera.table_z)])
        if ok.all():
            cv2.arrowedLine(overlay, tuple(np.rint(uv[0]).astype(int)), tuple(np.rint(uv[1]).astype(int)),
                            (255, 100, 20), 2)
    cv2.imwrite(str(output / 'mask.png'), mask * 255)
    cv2.imwrite(str(output / 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    # A real photo needs an independent geometric check. Project the known
    # desk outline for comparison with the photographed physical desk edges.
    # Pixel -> plane -> pixel roundtrip alone cannot validate extrinsics.
    calibration_overlay = rgb.copy()
    low, high = camera.table_bounds
    corners = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])
    xyz = np.c_[corners, np.full(4, camera.table_z)]
    uv, visible = camera.project(xyz)
    depth = ((xyz - camera.T_world_color[:3, 3]) @ camera.T_world_color[:3, :3])[:, 2]
    for i in range(4):
        j = (i + 1) % 4
        if depth[i] > 0 and depth[j] > 0:
            a, b = np.rint(uv[[i, j]]).astype(np.int32)
            ok, a, b = cv2.clipLine((0, 0, *camera.size), tuple(a), tuple(b))
            if ok:
                cv2.line(calibration_overlay, a, b, (50, 230, 80), 2)
        if visible[i]:
            cv2.putText(calibration_overlay, f'T{i}', tuple(np.rint(uv[i]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (50, 230, 80), 2)
    cv2.imwrite(str(output / 'calibration_overlay.png'), cv2.cvtColor(calibration_overlay, cv2.COLOR_RGB2BGR))
    np.savez_compressed(output / 'wet_region.npz', grid=grid, history=merged_history, origin=origin, cell=cell)
    export_wet_region(output / 'wet_region.usda', grid, origin, cell, camera.table_z, frame)
    record = dict(source_image=str(image.resolve()), image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                  frame=frame, units={'length': 'm', 'angle': 'rad'},
                  camera_pose_source=source, K=camera.K.tolist(), distortion=camera.distortion.tolist(),
                  image_size=list(camera.size), image_domain=image_domain, T_frame_color=camera.T_world_color.tolist(),
                  T_frame_robot_root=root.tolist(), T_robot_root_frame=inverse.tolist(),
                  table_z_m=camera.table_z, grid_cell_m=cell, detected_pixels=int(mask.sum()),
                  coarse_pixels=int(coarse.sum()), area_m2=float(grid.sum() * cell ** 2),
                  region_bounds_xy_m=bounds.tolist() if bounds is not None else None,
                  planning_region_center_xy_m=bounds.mean(0).tolist() if bounds is not None else None,
                  planning_margin_xy_m=list(margin), planning_footprint_xy_m=list(brush_size),
                  polygons_xy=[p.tolist() for p in polygons], polygons_robot_root_xyz=[p.tolist() for p in polygons_base],
                  candidate_contact_strokes_xy=[s.tolist() for s in strokes],
                  first_contact_stroke_xy=strokes[0].tolist() if strokes else None,
                  numerical_roundtrip_error_px=pixel_error,
                  calibration_overlay='calibration_overlay.png; compare the projected outline with real table edges',
                  volume_and_thickness_observed=False, hardware_localization_accuracy_validated=False,
                  trajectory_stage='Footprint and Cartesian contact candidates for sim planning; joint trajectory not exported',
                  detector_scope='Tinted/near-clear water on neutral tabletop; CAD occlusion and image-derived history',
                  inputs='External RGB + calibration + exposure-time camera pose + known table geometry')
    (output / 'observation.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
