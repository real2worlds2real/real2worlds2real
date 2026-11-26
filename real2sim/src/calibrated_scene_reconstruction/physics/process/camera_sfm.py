"""Independent process-video cameras from source-reviewed stationary regions.

Run in the local COLMAP environment. This produces a nonmetric camera/map
proposal, not a physical calibration or an accepted clock alignment. Camera
groups are explicit: sharing an event never implies sharing an intrinsic model.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .contract import digest, require, write


def prepare(records, camera_priors, output, *, width=1920):
    """Copy canonical images and independently reviewed static-region masks.

    Masks must come from source evidence, not from a projected candidate asset.
    Resizing preserves pixel-centre conventions for both source and intrinsics.
    """
    require(type(width) is int and width > 0, 'Positive image width required')
    require(records and set(camera_priors) == {r['camera'] for r in records},
            'Explicit independent intrinsics for every source camera required')
    output = Path(output)
    require(not output.exists(), 'Use a fresh camera proposal directory')
    checked = []
    for row in records:
        require(row.get('reviewed_static') is True and row.get('reason'),
                'Source-reviewed stationary support and an exclusion reason required')
        source, mask = Path(row['source']), Path(row['mask'])
        require(source.is_file() and mask.is_file()
                and digest(source) == row.get('source_sha256')
                and digest(mask) == row.get('mask_sha256'), 'Camera source or static mask changed')
        image = cv2.imread(str(source))
        allowed = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        require(image is not None and allowed is not None and allowed.shape == image.shape[:2]
                and np.any(allowed), 'Nonempty full-frame static mask required')
        prior = camera_priors[row['camera']]
        require(list(prior['size']) == [image.shape[1], image.shape[0]]
                and prior.get('basis'), 'Camera initializer needs its own image size and basis')
        K = np.asarray(prior['K'], float)
        require(K.shape == (3, 3) and np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0
                and np.allclose(K[2], [0, 0, 1]) and K[0, 1] == K[1, 0] == 0,
                'Finite positive pinhole intrinsic initializer required')
        require(np.isfinite(row['time_s']), 'Source video timestamps required')
        checked.append((row, image, allowed, K))
    output.mkdir(parents=True)
    saved, groups = [], {}
    for i, (row, image, allowed, K) in enumerate(checked):
        height = round(image.shape[0]*width/image.shape[1])
        sx, sy = width/image.shape[1], height/image.shape[0]
        scale = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
        intrinsic = scale@K
        name = f'{i:05d}.png'
        for directory in ['images', 'masks']:
            (output/directory).mkdir(exist_ok=True)
        require(cv2.imwrite(str(output/'images'/name), cv2.resize(image, (width, height))),
                'Could not save camera source image')
        mask_name = name+'.png'  # COLMAP image-name + .png convention.
        require(cv2.imwrite(str(output/'masks'/mask_name), cv2.resize(
            allowed, (width, height), interpolation=cv2.INTER_NEAREST)), 'Could not save static mask')
        group = groups.setdefault(row['camera'], {'K': intrinsic.tolist(), 'size': [width, height],
                                                   'basis': camera_priors[row['camera']]['basis'], 'names': []})
        require(np.allclose(group['K'], intrinsic) and group['size'] == [width, height],
                'A camera group cannot mix different source resolutions or intrinsics')
        group['names'].append(name)
        saved.append({**row, 'image': name, 'prepared_sha256': digest(output/'images'/name),
                      'prepared_mask_sha256': digest(output/'masks'/mask_name)})
    report = {'records': saved, 'groups': groups, 'accepted': False,
              'scale': 'arbitrary; independent metric robot registration required',
              'scope': 'Static-source-only reconstruction; no moving target, force or measured object dimensions.'}
    write(output/'inputs.json', report)
    return report


def reconstruct(directory, *, refine_intrinsics=False, workers=8):
    """COLMAP uses independent per-group cameras and per-frame poses.

    Fixed-intrinsic and self-calibrating proposals remain separately auditable;
    sparse reprojection alone does not validate metric scale or camera bias.
    """
    import pycolmap

    directory = Path(directory)
    inputs = json.loads((directory/'inputs.json').read_text())
    require(not (directory/'database.db').exists(), 'Use a fresh prepared proposal')
    for row in inputs['records']:
        require(digest(directory/'images'/row['image']) == row['prepared_sha256']
                and digest(directory/'masks'/(row['image']+'.png')) == row['prepared_mask_sha256'],
                'Prepared source evidence changed')
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = workers
    extraction.max_image_size = 1920
    extraction.sift.max_num_features = 12000
    database = directory/'database.db'
    for group in inputs['groups'].values():
        K = np.asarray(group['K'])
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = 'PINHOLE'
        reader.camera_params = ','.join(map(str, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
        reader.mask_path = directory/'masks'
        pycolmap.extract_features(database, directory/'images', image_names=group['names'],
                                  camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extraction)
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = workers
    pycolmap.match_exhaustive(database, matching_options=matching)
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = workers
    options.min_model_size = 6
    options.ba_refine_focal_length = refine_intrinsics
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.mapper.abs_pose_refine_focal_length = refine_intrinsics
    options.mapper.abs_pose_refine_extra_params = False
    models = pycolmap.incremental_mapping(database, directory/'images', directory/'sparse', options=options)
    summaries = [{'model': int(k), 'images': r.num_reg_images(), 'points': r.num_points3D(),
                  'mean_reprojection_error_px': r.compute_mean_reprojection_error(),
                  'cameras': {str(i): c.params.tolist() for i, c in r.cameras.items()}}
                 for k, r in models.items()]
    result = {'status': 'proposal' if summaries else 'no_static_model',
              'models': summaries, 'refine_intrinsics': refine_intrinsics, 'accepted': False,
              'metric_registration': 'pending', 'clock_alignment': 'pending',
              'input_sha256': digest(directory/'inputs.json')}
    write(directory/'result.json', result)
    return result
