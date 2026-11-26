"""Encode a native-state replay with visible per-frame quality and phase labels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def encode(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    for path, expected in manifest['sha256'].items():
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError('Recorded replay input changed: '+path)
    quality = json.loads(Path(manifest['quality']).read_text())
    with np.load(manifest['trace']) as trace:
        times, phases = trace['times'], trace['phases']
    if len(times) != len(quality) or not np.allclose(np.diff(times), 1/manifest['fps'], atol=1e-8):
        raise ValueError('Replay needs all uniformly sampled native states; no time interpolation')
    render = Path(manifest['output'])
    labeled = render/'labeled'
    labeled.mkdir(exist_ok=False)
    font = ImageFont.truetype(manifest.get('font', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), 19)
    failed = False
    for i, (time, phase, item) in enumerate(zip(times, phases, quality)):
        frame = Image.open(render/f'frame_{i:04d}.png').convert('RGB')
        draw = ImageDraw.Draw(frame)
        failed |= not item['frame_geometry_pass']
        draw.rectangle((0, 0, frame.width, 64), fill=(35, 10, 10) if failed else (15, 25, 35))
        status = 'Native simulation - PHYSICS CHECKS FAILED' if failed else 'Native simulation - candidate, task not validated'
        draw.text((12, 8), status, font=font, fill=(255, 225, 215) if failed else (245, 245, 245))
        draw.text((12, 35), f'{time:.2f}s   {str(phase).replace("_", " ")}   lid {np.degrees(item["lid_angle_rad"]):.1f} deg',
                  font=font, fill=(245, 245, 245))
        if manifest.get('preview_note'):
            draw.rectangle((0,frame.height-38,frame.width,frame.height),fill=(35,10,10))
            note_font=ImageFont.truetype(manifest.get('font','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),15)
            draw.text((12,frame.height-28),manifest['preview_note'],font=note_font,fill=(255,225,215))
        if manifest.get('material_grid'):
            draw.rectangle((0, frame.height-32, frame.width, frame.height), fill=(20, 25, 30))
            draw.text((12, frame.height-26), 'Blue lines: material coordinates on recorded cloth (display only).', font=font, fill=(225, 235, 245))
        frame.save(labeled/f'frame_{i:04d}.png')
    output = Path(manifest['video'])
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-framerate', str(manifest['fps']),
                    '-i', str(labeled/'frame_%04d.png'), '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(output)], check=True)
    (render/'video_scope.json').write_text(json.dumps({'video': str(output), 'native_states_only': True,
                                                      'physical_failure_visible': failed, 'task_approved': False,
                                                      'frames': len(times), 'fps': manifest['fps']}, indent=2)+'\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    print(encode(parser.parse_args().manifest))
