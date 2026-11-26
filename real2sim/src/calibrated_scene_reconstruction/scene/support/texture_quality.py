"""Flag compact high-contrast marks before repeating a captured material patch.

This only selects another observed patch; it never erases source pixels. A mark
may be real hardware or a material feature, so the result is not an object label.
"""
import cv2
import numpy as np


def periodic_tile(image):
    """Remove the smooth boundary mismatch before repeating observed material.

    A periodic-plus-smooth decomposition preserves the mean and fine grain.
    This is an inferred material continuation, never a replacement source image.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('A uint8 colour source patch is required')
    source = image.astype(float); h, w = source.shape[:2]
    boundary = np.zeros_like(source)
    boundary[0] += source[-1]-source[0]
    boundary[-1] += source[0]-source[-1]
    boundary[:, 0] += source[:, -1]-source[:, 0]
    boundary[:, -1] += source[:, 0]-source[:, -1]
    denominator = 2*np.cos(2*np.pi*np.arange(h)/h)[:, None]+2*np.cos(2*np.pi*np.arange(w)/w)[None, :]-4
    denominator[0, 0] = 1
    spectrum = np.fft.fft2(boundary, axes=(0, 1))/denominator[..., None]
    spectrum[0, 0] = 0
    smooth = np.fft.ifft2(spectrum, axes=(0, 1)).real
    result = np.rint(source-smooth).clip(0, 255).astype(np.uint8)
    def jump(a):
        a = a.astype(float)
        return float((np.abs(a[0]-a[-1]).mean()+np.abs(a[:, 0]-a[:, -1]).mean())/2)
    return result, {'method': 'periodic plus smooth boundary decomposition',
                    'source_boundary_jump_rgb': jump(image), 'tile_boundary_jump_rgb': jump(result),
                    'maximum_colour_correction_rgb': float(np.abs(smooth).max()),
                    'scope': 'Inferred repeating material; original observed crop retained separately.'}


def repeatability(image):
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('An observed uint8 BGR material patch is required')
    gray=cv2.cvtColor(cv2.resize(image,(128,128)),cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    residual=gray-cv2.GaussianBlur(gray,(0,0),4.)
    anomalies=((residual<-.18)|(residual>.22)).astype(np.uint8)
    count,_,stats,_=cv2.connectedComponentsWithStats(anomalies)
    compact=[]
    for x,y,w,h,area in stats[1:]:
        if 3<=area<=gray.size*.03 and max(w,h)<=3*max(1,min(w,h)):
            compact.append({'xywh':[int(x),int(y),int(w),int(h)],'area':int(area)})
    return {'suitable_for_repetition':not compact,'compact_contrast_marks':compact,
            'scope':'Source-patch selection only; retain unique marks separately when visible, do not tile them over the support'}
