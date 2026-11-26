"""Create indexed contact sheets covering every retained rendered movie frame."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(frames, output, fps, video=None, frame_offset=0):
    frames, output = Path(frames), Path(output)
    paths = sorted(frames.glob('frame_*.png'))
    if not paths or fps <= 0:
        raise ValueError('Supply rendered frame_*.png images and a positive FPS')
    numbers = [int(path.stem.split('_')[-1]) for path in paths]
    if numbers != list(range(numbers[0], numbers[0]+len(numbers))):
        raise ValueError('Rendered movie frames have gaps')
    output.mkdir(parents=True, exist_ok=True)
    sheets, entries = [], []
    for start in range(0, len(paths), 12):
        page = Image.new('RGB', (1536, 936), '#eeeeee')
        draw = ImageDraw.Draw(page)
        for local, path in enumerate(paths[start:start+12]):
            index = start+local
            with Image.open(path) as image:
                thumb = ImageOps.contain(image.convert('RGB'), (384, 288))
                x, y = local % 4 * 384, local // 4 * 312
                page.paste(thumb, (x+(384-thumb.width)//2, y))
            draw.text((x+5, y+290), f'frame {index} | {index/fps:.3f}s | cache {index+frame_offset}', fill='black')
            entries.append({'frame': index, 'time_seconds': index/fps, 'cache_frame': index+frame_offset,
                            'path': str(path.resolve()), 'sha256': digest(path), 'sheet': f'sheet_{start//12:03d}.jpg'})
        target = output/f'sheet_{start//12:03d}.jpg'
        page.save(target, quality=93)
        sheets.append(target.name)
    report = {'status': 'awaiting_visual_review', 'coverage': 1.0, 'frames': len(paths), 'fps': fps,
              'representation': 'Every retained rendered frame as indexed contact sheets; originals retained for close inspection. This is not a claim of native video playback or encoded-frame equality.',
              'sheets': sheets, 'frame_index': entries}
    if video is not None:
        report['video'] = {'path': str(Path(video).resolve()), 'sha256': digest(Path(video))}
    (output/'index.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frames', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--frame-offset', type=int, default=0)
    args = parser.parse_args()
    report = prepare(args.frames, args.output, args.fps, args.video, args.frame_offset)
    print('MOTION_REVIEW_PREPARED', report['frames'], 'frames', len(report['sheets']), 'sheets')


if __name__ == '__main__':
    main()
