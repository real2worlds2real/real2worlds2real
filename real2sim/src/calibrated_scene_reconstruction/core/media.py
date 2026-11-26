"""Shared video decoding and color adaptation for scene and physics observations.

Frame indices refer to decoded source frames, never a resampled proxy. HDR uses
the compatible HLG/PQ base layer, not Dolby Vision dynamic display metadata.
"""
import hashlib
import json
import math
from numbers import Integral
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def _identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def _json(command):
    return json.loads(subprocess.check_output(command, stderr=subprocess.PIPE))


@lru_cache(maxsize=32)
def _probe(path, size, mtime):
    data = _json(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                  '-show_streams', '-of', 'json', path])
    if not data.get('streams'):
        raise ValueError('No video stream: '+path)
    return data['streams'][0]


def probe(path):
    """Read source color/rotation metadata; unknown metadata is not guessed HDR."""
    return _probe(*_identity(path))


def frame_timestamp(path, frame):
    """Presentation time of the decoded source frame (also for VFR inputs)."""
    return _timestamps(*_identity(path))[frame]


@lru_cache(maxsize=32)
def _timestamps(path, size, mtime):
    data = _json(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                  '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', path])
    return tuple(float(row['best_effort_timestamp_time']) for row in data['frames'])


@lru_cache(maxsize=1)
def _versions():
    return {name: subprocess.check_output([name, '-version'], text=True).splitlines()[0]
            for name in ('ffmpeg', 'ffprobe')}


def color_profile(path, exposure_ev=0.):
    """Deterministic clip-wide conversion, suitable for photometric comparison.

    Exposure variants are review evidence, not independent observations/cameras.
    SDR defaults to the existing OpenCV path without contrast normalization.
    """
    if not math.isfinite(exposure_ev) or abs(exposure_ev) > 8:
        raise ValueError('Exposure adjustment must be finite and within 8 stops')
    stream = probe(path)
    transfer = stream.get('color_transfer')
    hdr = transfer in ('arib-std-b67', 'smpte2084')
    dovi = next((row for row in stream.get('side_data_list', [])
                 if 'DOVI' in row.get('side_data_type', '')), None)
    if dovi and (not hdr or dovi.get('bl_present_flag') != 1
                 or (dovi.get('dv_bl_signal_compatibility_id'), transfer)
                 not in ((1, 'smpte2084'), (4, 'arib-std-b67'))):
        raise ValueError('Unsupported Dolby Vision base layer; supply a color-managed conversion')
    if hdr and (stream.get('color_primaries') not in ('bt2020', 'bt709')
                or stream.get('color_space') not in ('bt2020nc', 'bt709')):
        raise ValueError('HDR needs explicit supported color primaries and matrix metadata')
    if not hdr and exposure_ev:
        raise ValueError('SDR exposure brackets cannot recover discarded source highlights')
    peak = 10 if transfer == 'arib-std-b67' else 100
    filters = None
    if hdr:
        filters = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,'
                   f'exposure=exposure={exposure_ev:g},tonemap=hable:desat=2:peak={peak},'
                   'zscale=t=iec61966-2-1:m=gbr:r=full,format=bgr24')
    return {'schema': 1, 'mode': 'hdr_base_to_srgb' if hdr else 'opencv_sdr',
            'source': {key: stream.get(key) for key in
                       ('pix_fmt', 'color_range', 'color_space', 'color_transfer', 'color_primaries')},
            'side_data': stream.get('side_data_list', []), 'exposure_ev': exposure_ev,
            'filter': filters, 'versions': {**_versions(), 'opencv': cv2.__version__},
            'engine_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitation': 'Dolby Vision dynamic RPU metadata is not applied' if dovi else None}


def exposure_metrics(bgr):
    """Diagnostic only: bright white may be real material, not clipped source."""
    pixels = np.asarray(bgr)
    return {'near_white_fraction': float(np.all(pixels >= 245, axis=-1).mean()),
            'any_channel_at_ceiling_fraction': float(np.any(pixels >= 254, axis=-1).mean()),
            'white_at_ceiling_fraction': float(np.all(pixels >= 254, axis=-1).mean()),
            'near_black_fraction': float(np.all(pixels <= 5, axis=-1).mean()),
            'source_clipping_determined': False,
            'interpretation': 'Review task boundaries and texture locally; global brightness cannot certify visibility'}


class VideoCapture:
    """Small OpenCV-compatible reader with explicit HDR adaptation and exact seek.

    Only FPS/count/size/position properties are supported. HDR seeking decodes
    from the source start and selects frame n; phone keyframe seeking is avoided.
    """
    def __init__(self, path, *, exposure_ev=0.):
        self.path = str(path)
        self.profile = color_profile(path, exposure_ev)
        self.hdr = self.profile['mode'] != 'opencv_sdr'
        self.cap = cv2.VideoCapture(self.path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.hdr:
            stream = probe(path)
            rotation = next((row['rotation'] for row in stream.get('side_data_list', [])
                             if 'rotation' in row), 0)
            if abs(rotation/90-round(rotation/90)) > 1e-6:
                self.cap.release()
                raise ValueError('HDR decoder requires a quarter-turn display rotation')
            self.width, self.height = stream['width'], stream['height']
            if round(rotation/90) % 2:
                self.width, self.height = self.height, self.width
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.opened = self.cap.isOpened()
        self.position = 0
        self.last = None
        self.proc = self.errors = None
        if self.hdr:
            self.cap.release()

    def isOpened(self):
        return self.opened

    def _start(self):
        self.errors = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-i', self.path,
             '-map', '0:v:0', '-an', '-sn', '-vf',
             f'select=gte(n\\,{self.position}),'+self.profile['filter'],
             '-fps_mode', 'passthrough', '-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
            stdout=subprocess.PIPE, stderr=self.errors)

    def _stop(self):
        if self.proc is not None:
            self.proc.stdout.close()
            if self.proc.poll() is None:
                self.proc.terminate()
            self.proc.wait()
            self.errors.close()
            self.proc = self.errors = None

    def read(self):
        if not self.opened:
            return False, None
        if not self.hdr:
            return self.cap.read()
        if self.proc is None:
            self._start()
        needed = self.width*self.height*3
        data = self.proc.stdout.read(needed)
        if len(data) != needed:
            code = self.proc.wait()
            self.errors.seek(0)
            error = self.errors.read().decode(errors='replace')
            self._stop()
            self.opened = False
            if code != 0 or data:
                raise RuntimeError('HDR frame decode failed: '+error)
            return False, None
        self.position += 1
        return True, np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3).copy()

    def get(self, prop):
        if not self.hdr:
            return self.cap.get(prop)
        if prop == cv2.CAP_PROP_POS_MSEC:
            stamps = _timestamps(*_identity(self.path))
            return 1000*stamps[max(0, self.position-1)] if stamps else 0.
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.count,
                cv2.CAP_PROP_FRAME_WIDTH: self.width, cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                cv2.CAP_PROP_POS_FRAMES: self.position}.get(prop, 0.)

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES or not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError('Only nonnegative integer source frame seeks are supported')
        if not self.hdr:
            # Exact sequential seek also for SDR phone HEVC.
            if value < self.cap.get(cv2.CAP_PROP_POS_FRAMES):
                self.cap.release()
                self.cap = cv2.VideoCapture(self.path)
            while self.cap.get(cv2.CAP_PROP_POS_FRAMES) < value:
                if not self.cap.grab():
                    return False
            return True
        if value != self.position:
            self._stop()
            self.position = int(value)
        self.opened = True
        return True

    def grab(self):
        if not self.hdr:
            return self.cap.grab()
        ok, self.last = self.read()
        return ok

    def retrieve(self):
        return (self.last is not None, self.last) if self.hdr else self.cap.retrieve()

    def release(self):
        self._stop()
        self.cap.release()
        self.opened = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def extract_frames(path, indices, *, exposure_ev=0.):
    """Batch exact frame selection before expensive HDR conversion; no resampling."""
    wanted = sorted(set(indices))
    if not wanted or any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 for i in wanted):
        raise ValueError('Select nonnegative integer frame indices')
    with VideoCapture(path, exposure_ev=exposure_ev) as reader:
        if not reader.hdr:
            result = {}
            for i in wanted:
                reader.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, image = reader.read()
                if not ok:
                    raise ValueError(f'Source frame {i} is unavailable')
                result[i] = image
            return result
        select = '+'.join(f'eq(n\\,{i})' for i in wanted)
        with tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(
                ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-i', str(path),
                 '-map', '0:v:0', '-an', '-sn', '-vf', 'select='+select+','+reader.profile['filter'],
                 '-frames:v', str(len(wanted)), '-fps_mode', 'passthrough', '-threads', '2',
                 '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE, stderr=errors)
            result = {}
            try:
                for i in wanted:
                    data = proc.stdout.read(reader.width*reader.height*3)
                    if len(data) != reader.width*reader.height*3:
                        raise RuntimeError(f'Incomplete source frame {i}')
                    result[i] = np.frombuffer(data, np.uint8).reshape(reader.height, reader.width, 3).copy()
                if proc.wait() != 0:
                    raise RuntimeError('HDR batch decode failed')
            finally:
                proc.stdout.close()
                if proc.poll() is None:
                    proc.terminate()
                proc.wait()
            return result
