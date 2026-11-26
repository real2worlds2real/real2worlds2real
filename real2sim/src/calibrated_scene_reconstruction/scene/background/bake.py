"""Projective texture baking of planar rectangles from the photo (with per-texel 'observed' masks)."""
import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.inpaint import removed_mask


class ProjectionContext:
    """Photo geometry in the world frame: project world points, test visibility, sample a source image."""

    def __init__(self, s: Session):
        geo = s.geometry()
        self.P, self.valid, self.depth = geo["points"], geo["mask"], geo["depth"]
        self.K = geo["K"].astype(np.float64)
        self.masks = s.masks()
        self.H, self.W = self.depth.shape
        self.R_wc, self.o, self.T_wc = s.world_frame()
        self.photo_pose_source = 'bootstrap photo geometry'
        posed = s / 'parsing/posed_frames/cameras.npz'
        keyframes = s / 'capture/keyframes.json'
        if posed.exists() and keyframes.exists():
            import json
            photo = f"0_{json.loads(keyframes.read_text())['photo']:05d}.jpg"
            with np.load(posed) as cameras:
                names = list(map(str, cameras['names']))
                if photo not in names:
                    raise ValueError('Reference photo missing from registered cameras; cannot bake a posed scene')
                i = names.index(photo)
                self.K = cameras['K'][i].astype(float).copy()
                width, height = cameras['size'][i]
                self.K[0] *= self.W / width
                self.K[1] *= self.H / height
                self.R_wc = cameras['R_world_cam'][i].astype(float)
                centre = cameras['centre'][i].astype(float)
            self.o = -self.R_wc.T @ centre
            self.T_wc = np.eye(4)
            self.T_wc[:3, :3], self.T_wc[:3, 3] = self.R_wc, centre
            yy, xx = np.indices(self.depth.shape)
            rays = np.stack([(xx-self.K[0, 2])/self.K[0, 0],
                             (yy-self.K[1, 2])/self.K[1, 1], np.ones_like(xx)], axis=-1)
            # Bootstrap depth has its earlier provisional scale. Use the same
            # capture's observed floor and calibrated camera height to adapt it
            # for texture visibility, without mutating metric fitting inputs.
            floor_ids = s.scene_graph.get('structure', {}).get('floor', {}).get('ids', [])
            floor = np.any(self.masks[floor_ids], axis=0) if floor_ids else np.zeros_like(self.valid)
            height_term = self.depth * (rays @ self.R_wc[2])
            known = floor & self.valid & np.isfinite(height_term) & (height_term < -1e-6)
            scale = -float(centre[2]) / float(np.median(height_term[known])) if known.sum() >= 100 else 1.0
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid photo-depth adaptation from the observed floor')
            self.depth = self.depth * scale
            self.P = rays * self.depth[..., None]
            self.photo_pose_source = {'registered_photo': photo, 'depth_scale_for_texture_visibility': scale,
                                      'floor_pixels': int(known.sum()), 'metric_fitting_inputs_modified': False}
        self.removed = removed_mask(s)
        self.source = None  # float32 HxWx3 image the textures are sampled from
        self.pano = None  # PanoSource: completed views of what the photo did not see (recon.panorama)
        self.pano_gain = np.ones(3, np.float32)  # pano radiance -> this source (albedo: photo albedo / photo ratio)

    def union(self, ids):
        if "*" in ids:  # any pixel; visibility decided by the depth test alone
            return self.valid.copy()
        m = np.zeros((self.H, self.W), bool)
        for i in ids:
            m |= self.masks[i]
        return m

    def to_world(self, p_c):
        return (p_c - self.o) @ self.R_wc.T

    def project(self, p_w):
        pc = p_w @ self.R_wc + self.o
        z = pc[..., 2]
        u = pc[..., 0] / np.maximum(z, 1e-6) * self.K[0, 0] + self.K[0, 2]
        v = pc[..., 1] / np.maximum(z, 1e-6) * self.K[1, 1] + self.K[1, 2]
        return u, v, z


def bake_rect(ctx: ProjectionContext, O, U, V, su, sv, res, ids, fill_color=None, depth_tol=0.12):
    """Texture for the rectangle O + a*U + b*V (a in [0,su], b in [0,sv]); row 0 of the texture is b = 0.

    A texel is *observed* if it projects inside the photo onto an allowed instance (or an inpainted object pixel)
    whose MoGe depth agrees with the rectangle; with ctx.pano, unobserved texels the completed views see (depth
    agreeing, or a removed generated object) are *covered* and take the view plate's colour; the rest get the observed
    mean (or fill_color) plus noise. Returns (texture, observed, corners, covered).
    """
    nu, nv = max(2, int(su / res)), max(2, int(sv / res))
    A, B = np.meshgrid((np.arange(nu) + 0.5) / nu * su, (np.arange(nv) + 0.5) / nv * sv)
    pts = O + A[..., None] * U + B[..., None] * V
    u, v, z = ctx.project(pts)
    inside = (z > 0.1) & (u >= 0) & (u < ctx.W - 1) & (v >= 0) & (v < ctx.H - 1)
    ui = np.clip(np.round(u).astype(int), 0, ctx.W - 1)
    vi = np.clip(np.round(v).astype(int), 0, ctx.H - 1)
    allowed = ctx.union(ids) | ctx.removed
    depth_ok = np.abs(ctx.depth[vi, ui] - z) < depth_tol * z
    obs = inside & allowed[vi, ui] & (depth_ok | ctx.removed[vi, ui]) & ctx.valid[vi, ui]
    tex = cv2.remap(ctx.source, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    cov = np.zeros_like(obs)
    if ctx.pano is not None:
        col, cov = ctx.pano.sample(pts, fallback=abs(np.cross(U, V)[2]) < 0.5)  # walls: no holes behind occluders
        cov &= ~obs
    if obs.sum() < 50 and cov.sum() < 50:
        tex[:] = np.asarray(fill_color if fill_color is not None else [200, 200, 200], np.float32)
    else:
        ref = obs if obs.sum() >= 50 else cov
        src = np.where(cov[..., None], col * ctx.pano_gain, tex) if cov.any() else tex
        mean = src[ref].mean(0) if fill_color is None else np.asarray(fill_color, np.float32)
        noise = np.random.default_rng(0).normal(0, 3, tex.shape).astype(np.float32)
        tex = np.where((obs | cov)[..., None], src, mean + noise)
    corners = np.array([O, O + su * U, O + su * U + sv * V, O + sv * V])
    return tex.clip(0, 255).astype(np.uint8), obs, corners, cov


class PanoSource:
    """Completed views around the photo camera (recon.panorama): object-free plates, metric depth, cameras."""

    def __init__(self, d):
        z = np.load(d / "views.npz")
        self.names = [str(n) for n in z["names"]]
        self.R, self.K, self.cam = z["R"], z["K"], z["cam"]
        self.depth, self.ok, self.known = z["depth"], z["ok"], z["known"]
        self.plates = [np.asarray(Image.open(d / "views" / f"{n}_plate.png").convert("RGB"), np.float32)
                       for n in self.names]
        self.removed = [np.asarray(Image.open(d / "views" / f"{n}_mask.png").convert("L")) > 127 for n in self.names]

    def sample(self, pts, near_frac=0.4, fallback=False):
        """Colour and coverage of world points (..., 3): the most frontal view that sees them unoccluded. With
        fallback (walls), points every view sees behind a generated occluder (a door frame, a pillar) take the most
        frontal view anyway: the occluder painted on the wall reads better than a patch of another material."""
        d = pts - self.cam
        dist = np.linalg.norm(d, axis=-1)
        u_ = d / np.maximum(dist, 1e-6)[..., None]
        col = np.zeros(pts.shape, np.float32)
        cov = np.zeros(pts.shape[:-1], bool)
        best = np.full(pts.shape[:-1], -np.inf)
        col_any, best_any = np.zeros(pts.shape, np.float32), np.full(pts.shape[:-1], -np.inf)
        for i in range(len(self.names)):
            dc = u_ @ self.R[i]
            z = dc[..., 2]
            K = self.K[i] if self.K.ndim == 3 else self.K
            u = dc[..., 0] / np.maximum(z, 1e-6) * K[0, 0] + K[0, 2]
            v = dc[..., 1] / np.maximum(z, 1e-6) * K[1, 1] + K[1, 2]
            h, w = self.depth[i].shape
            ui, vi = np.clip(u.astype(int), 0, w - 1), np.clip(v.astype(int), 0, h - 1)
            # the photo's own pixels are the photo bake's business (there they show its objects, not the surface)
            seen = (z > 0.2) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & ~self.known[i][vi, ui]
            # depth from a generated view is loose: only a clear occluder in front of the surface (not removed by the
            # object masks) keeps the view from painting it
            occluded = self.ok[i][vi, ui] & (self.depth[i][vi, ui] < (1 - near_frac) * dist) & ~self.removed[i][vi, ui]
            good = seen & ~occluded & (z > best)
            anyv = seen & (z > best_any) if fallback else np.zeros_like(good)
            if not (good.any() or anyv.any()):
                continue
            c = cv2.remap(self.plates[i], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
            col[good], cov[good], best[good] = c[good], True, z[good]
            col_any[anyv], best_any[anyv] = c[anyv], z[anyv]
        if fallback:
            fb = ~cov & np.isfinite(best_any)
            col[fb], cov[fb] = col_any[fb], True
        return col, cov


class PosedSource:
    """Posed walk frames (recon.posed_frames: COLMAP at the board's scale, a MoGe-2 depth each) as PanoSource's stand-in:
    real frames from all round the room for the structure planes the photo does not see. A texel takes the most
    frontal frame that sees it unoccluded -- the frame's depth there within `near_frac` of the texel's distance (a desk
    in front of a wall keeps that frame from painting it onto the wall) -- and no fallback: a wall every frame sees
    only behind furniture stays unobserved (the furniture is a room object of its own, its picture on the wall a
    ghost). Each frame's colour is scaled to the median of all the frames' (the walks' exposure drifts)."""

    def __init__(self, path, near_frac=0.1):
        z = np.load(path)
        self.names = [str(n) for n in z["names"]]
        self.R, self.K, self.cam = z["R_world_cam"], z["K"], z["centre"]
        self.depth, self.ok = z["depth"].astype(np.float32), z["ok"]
        self.plates = [im.astype(np.float32) for im in z["images"]]
        lum = np.array([np.median(p[ok].mean(-1)) for p, ok in zip(self.plates, self.ok)])
        self.gain = np.median(lum) / np.maximum(lum, 1.0)
        self.near_frac = near_frac
        self.known = [np.zeros(d.shape, bool) for d in self.depth]

    def sample(self, pts, near_frac=None, fallback=False):
        near = self.near_frac if near_frac is None else near_frac
        nrm = None
        if pts.ndim == 3 and min(pts.shape[:2]) > 1:  # a plane's texel grid: its normal, for the incidence
            nrm = np.cross(pts[0, 1] - pts[0, 0], pts[1, 0] - pts[0, 0])
            nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
        col = np.zeros(pts.shape, np.float32)
        cov = np.zeros(pts.shape[:-1], bool)
        best = np.full(pts.shape[:-1], -np.inf)
        for i in range(len(self.names)):
            d = pts - self.cam[i]
            dist = np.linalg.norm(d, axis=-1)
            dc = (d / np.maximum(dist, 1e-6)[..., None]) @ self.R[i]
            zc = dc[..., 2]
            K = self.K[i]
            u = dc[..., 0] / np.maximum(zc, 1e-6) * K[0, 0] + K[0, 2]
            v = dc[..., 1] / np.maximum(zc, 1e-6) * K[1, 1] + K[1, 2]
            h, w = self.depth[i].shape
            seen = (zc > 0.2) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (dist < 12)
            if nrm is not None:  # not at a grazing angle (a floor seen along it took the desks standing on it)
                seen &= np.abs(d @ nrm) > 0.25 * dist
            if not seen.any():
                continue
            ui, vi = np.clip(u.astype(int), 0, w - 1), np.clip(v.astype(int), 0, h - 1)
            planar = dist * zc  # the texel's depth along the frame's optical axis (the frames store planar depth)
            dd = self.depth[i][vi, ui]
            # within `near` of the depth and at most 0.25 m: at 5 m, 10 % let a desk half a metre in front paint a wall
            clear = self.ok[i][vi, ui] & (np.abs(dd - planar) < np.minimum(near * planar, 0.25))
            good = seen & clear & (zc > best)
            if not good.any():
                continue
            c = cv2.remap(self.plates[i], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE) * self.gain[i]
            col[good], cov[good], best[good] = c[good], True, zc[good]
        return col, cov


def smooth_fill(tex, obs, sigma):
    """Plain painted panels: keep only the low-frequency lighting (removes inpainting ghosts)."""
    filled = cv2.inpaint(tex, (~obs).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return cv2.GaussianBlur(filled.astype(np.float32), (0, 0), sigma).clip(0, 255).astype(np.uint8)
