"""RGB-only tabletop wiping geometry; no simulator state or particle inputs.

The deliberately simple detector is for blue-tinted water on a neutral desk.
All pixel coordinates use OpenCV optical axes and the supplied resolution's K.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TableCamera:
    K: np.ndarray
    distortion: np.ndarray
    size: tuple
    T_world_color: np.ndarray
    table_z: float
    table_bounds: np.ndarray

    def __post_init__(self):
        self.K = np.asarray(self.K, float)
        self.distortion = np.asarray(self.distortion, float)
        self.T_world_color = np.asarray(self.T_world_color, float)
        self.table_bounds = np.asarray(self.table_bounds, float)
        T = self.T_world_color
        if (self.K.shape != (3, 3) or T.shape != (4, 4)
                or not np.isfinite(self.K).all() or not np.isfinite(T).all()
                or not np.allclose(T[3], [0, 0, 0, 1])
                or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
                or np.linalg.det(T[:3, :3]) < .99):
            raise ValueError("Invalid calibrated camera transform/intrinsics")
        if (len(self.size) != 2 or min(self.size) <= 0 or self.K[0, 0] <= 0
                or self.K[1, 1] <= 0 or not np.isfinite(self.distortion).all()
                or self.table_bounds.shape != (2, 2)
                or not np.all(self.table_bounds[1] > self.table_bounds[0])
                or not np.isfinite(self.table_z)):
            raise ValueError("Invalid resolution, distortion or table extent")

    def check_image(self, rgb):
        if rgb.shape != (self.size[1], self.size[0], 3):
            raise ValueError(f"Image shape {rgb.shape} does not match calibration {self.size}")

    def pixel_to_table(self, pixels):
        pixels = np.asarray(pixels, float).reshape(-1, 2)
        if not len(pixels):
            return np.empty((0, 3)), np.zeros(0, bool)
        undistorted = cv2.undistortPoints(pixels[:, None], self.K, self.distortion)[:, 0]
        rays = np.c_[undistorted, np.ones(len(pixels))] @ self.T_world_color[:3, :3].T
        origin = self.T_world_color[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (self.table_z - origin[2]) / rays[:, 2]
            points = origin + distance[:, None] * rays
        valid = ((distance > 0) & (np.abs(rays[:, 2]) > 1e-6) & np.isfinite(points).all(1)
                 & (points[:, :2] >= self.table_bounds[0]).all(1)
                 & (points[:, :2] <= self.table_bounds[1]).all(1))
        return points, valid

    def project(self, points, distorted=True):
        points = np.asarray(points, float).reshape(-1, 3)
        pc = (points - self.T_world_color[:3, 3]) @ self.T_world_color[:3, :3]
        if not len(pc):
            return np.empty((0, 2)), np.zeros(0, bool)
        pixels = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), self.K,
                                  self.distortion if distorted else np.zeros(8))[0][:, 0]
        visible = ((pc[:, 2] > 0) & (pixels >= 0).all(1)
                   & (pixels < np.asarray(self.size)).all(1))
        return pixels, visible

    def distort_render(self, undistorted_rgb):
        """Synthesize raw lens distortion once; the detector then consumes RGB only."""
        self.check_image(undistorted_rgb)
        width, height = self.size
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        raw_pixels = np.stack([x, y], -1).astype(np.float32)
        source = cv2.undistortPoints(raw_pixels.reshape(-1, 1, 2), self.K,
                                     self.distortion, P=self.K).reshape(height, width, 2)
        return cv2.remap(undistorted_rgb, source[..., 0], source[..., 1], cv2.INTER_LINEAR)


def detect_blue_water(rgb, camera, minimum_pixels=12, *, history=None):
    camera.check_image(rgb)
    c = rgb.astype(np.int16)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Bright neutral reflections need not be blue. A small closing repairs those
    # gaps without assuming the simulation's actual puddle centre or outline.
    mask = ((c[..., 2] - c[..., 0] > 18) & (c[..., 1] - c[..., 0] > 5)
            & (hsv[..., 1] > 35) & (hsv[..., 0] >= 80) & (hsv[..., 0] <= 125))
    # A thin residual mixes with the tan desk and can become pale green/teal.
    # Accept that weaker evidence only near water observed in earlier photos.
    fine = ((c[..., 1] - c[..., 0] > 3) & (c[..., 2] - c[..., 0] > -15)
            & (hsv[..., 1] > 18) & (hsv[..., 1] < 125)
            & (hsv[..., 0] >= 45) & (hsv[..., 0] <= 130)) if history is not None else np.zeros_like(mask)
    if history is not None:
        # Subpixel water can inherit the brown desk's hue. Compare a small
        # trace with nearby dry desk instead of requiring blue/green pixels.
        # This weak evidence is still gated by image-derived wet history.
        background = cv2.medianBlur(rgb, 31).astype(np.float32)
        drop = background - c
        current_ratio = (c[..., 1:].astype(np.float32) + 1) / (c[..., :1] + 1)
        background_ratio = (background[..., 1:] + 1) / (background[..., :1] + 1)
        ratio_change = current_ratio - background_ratio
        fine |= ((drop[..., 0] > 12) & (drop[..., 1] > 6)
                 & (ratio_change[..., 0] > .020) & (ratio_change[..., 1] > .040)
                 & (c[..., 0] > .45 * background[..., 0]) & (hsv[..., 1] < 150))
    ys, xs = np.nonzero(mask | fine)
    points, valid = camera.pixel_to_table(np.c_[xs, ys])
    supported = np.zeros(len(xs), bool)
    if history is not None:
        grid, origin, cell = history
        if cell <= 0 or grid.ndim != 2:
            raise ValueError('Invalid observed water history')
        radius = int(np.ceil(.015 / cell))
        region = cv2.dilate(grid.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        ij = np.floor((points[valid, :2] - np.asarray(origin)) / cell).astype(int)
        inside = ((ij >= 0).all(1) & (ij < np.array(grid.shape[::-1])).all(1))
        ids = np.flatnonzero(valid)[inside]
        supported[ids] = region[ij[inside, 1], ij[inside, 0]] > 0
    clipped = np.zeros(mask.shape, np.uint8)
    accepted = valid & (mask[ys, xs] | (fine[ys, xs] & supported))
    clipped[ys[accepted], xs[accepted]] = 1
    small_supported = np.zeros_like(clipped)
    small_supported[ys[supported & accepted], xs[supported & accepted]] = 1
    clipped = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clipped)
    supported_labels = set(np.unique(labels[small_supported > 0])) - {0}
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum_pixels or i in supported_labels]
    return np.isin(labels, keep).astype(np.uint8)


def projected_robot_mask(camera, poses, local_vertices, padding=4):
    """Occlusion from known CAD/FK, usable for real photographs without labels."""
    mask = np.zeros((camera.size[1], camera.size[0]), np.uint8)
    for name, vertices in local_vertices.items():
        if name not in poses or len(vertices) < 3:
            continue
        transform = poses[name]
        xyz = np.asarray(vertices) @ transform[:3, :3].T + transform[:3, 3]
        depth = ((xyz - camera.T_world_color[:3, 3]) @ camera.T_world_color[:3, :3])[:, 2]
        uv, _ = camera.project(xyz[depth > .02])
        if len(uv) >= 3 and np.isfinite(uv).all():
            hull = cv2.convexHull(np.clip(np.rint(uv), -100000, 100000).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1)
    if padding:
        mask = cv2.dilate(mask, np.ones((padding*2+1, padding*2+1), np.uint8))
    return mask


def detect_table_water(rgb, camera, *, history=None, exclude_mask=None):
    """Colour plus local optical contrast, including near-colourless thin water.

    A robust spatial background is estimated from this photo only. An empty
    table reference, fluid coordinates and rendered segmentation are not inputs.
    """
    camera.check_image(rgb)
    base = detect_blue_water(rgb, camera, history=history)
    c = rgb.astype(np.float32)
    background = cv2.medianBlur(rgb, 127).astype(np.float32)
    drop = background - c
    ratio = (c[..., 1:] + 1) / (c[..., :1] + 1)
    reference_ratio = (background[..., 1:] + 1) / (background[..., :1] + 1)
    change = np.abs(ratio-reference_ratio)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    weak = ((drop[..., 0] > 7) & (drop[..., 1] > 4)
            & ((change[..., 0] > .010) | (change[..., 1] > .020))
            & (c[..., 0] > .45*background[..., 0]) & (hsv[..., 1] > 35)
            & (hsv[..., 1] < 155) & (hsv[..., 0] < 80))
    if exclude_mask is not None:
        if exclude_mask.shape != weak.shape:
            raise ValueError('Robot occlusion mask does not match the image')
        weak &= exclude_mask == 0
        base[exclude_mask > 0] = 0
    y,x = np.nonzero(weak)
    points,valid = camera.pixel_to_table(np.c_[x,y])
    clipped = np.zeros_like(base)
    clipped[y[valid],x[valid]] = 1
    clipped = cv2.morphologyEx(clipped,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    if exclude_mask is not None:
        clipped[exclude_mask > 0] = 0
    supported_labels = set()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clipped)
    if history is not None and valid.any():
        grid, origin, cell = history
        radius = int(np.ceil(.015/cell))
        support = cv2.dilate(grid.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))
        ij = np.floor((points[valid,:2]-np.asarray(origin))/cell).astype(int)
        inside = (ij>=0).all(1)&(ij<np.array(grid.shape[::-1])).all(1)
        ids = np.flatnonzero(valid)[inside]
        hits = support[ij[inside,1],ij[inside,0]] > 0
        supported_labels = set(labels[y[ids[hits]],x[ids[hits]]])-{0}
    # Shadows/held cloth that remain attached to the known robot silhouette
    # are ambiguous foreground, not independent puddle evidence.
    touching_robot = (set(labels[cv2.dilate((exclude_mask>0).astype(np.uint8),np.ones((5,5),np.uint8))>0])
                      if exclude_mask is not None else set())
    # A shadow can detach slightly from the CAD silhouette. Weak contrast in
    # that halo is insufficient evidence for a NEW puddle. Previously observed
    # wet traces remain eligible there; do not erase tiny historical remnants.
    shadow_radius = max(2, int(np.ceil(20 * camera.size[0] / 960)))
    near_robot = (set(labels[cv2.dilate((exclude_mask>0).astype(np.uint8),
                                      np.ones((2*shadow_radius+1,2*shadow_radius+1),np.uint8))>0])
                  if exclude_mask is not None else set())
    keep = []
    for i in range(1,count):
        w,h = stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT]
        compact = max(w,h)/max(min(w,h),1) <= 5
        if i not in touching_robot and (i in supported_labels or
                (i not in near_robot and stats[i,cv2.CC_STAT_AREA] >= 30 and compact)):
            keep.append(i)
    return np.maximum(base,np.isin(labels,keep).astype(np.uint8))


def mask_to_grid(mask, camera, cell=.002):
    """Project every accepted raw-image pixel into a metric tabletop occupancy grid."""
    lo, hi = camera.table_bounds
    shape = np.ceil((hi - lo) / cell).astype(int)[::-1]
    grid = np.zeros(shape, np.uint8)
    ys, xs = np.nonzero(mask)
    points, valid = camera.pixel_to_table(np.c_[xs, ys])
    points = points[valid]
    if not len(points):
        return grid, lo.copy(), cell
    xy = np.floor((points[:, :2] - lo) / cell).astype(int)
    xy = np.minimum(xy, np.array(shape[::-1]) - 1)
    grid[xy[:, 1], xy[:, 0]] = 1
    # A camera pixel covers several 2 mm cells at this range. Fill sampling holes
    # in metric space, then preserve separate connected puddles for planning.
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(grid, contours, -1, 1, cv2.FILLED)
    return grid, lo.copy(), cell


def coverage_strokes(grid, origin, cell, brush_size=(.055, .065), margin=.009,
                     overlap=.3, edge_margin=.025, axis=0, clip_to_table=False):
    """Parallel contact strokes from observed water only, with explicit edge rejection.

    Brush dimensions always use world XY. Axis chooses the travel direction.
    Every returned stroke is followed by a separate raised transfer in the app.
    """
    margin = np.broadcast_to(np.asarray(margin, float), (2,))
    if axis == 1:
        swapped = coverage_strokes(grid.T, np.asarray(origin)[::-1], cell, np.asarray(brush_size)[::-1],
                                   margin[::-1], overlap, edge_margin, axis=0,clip_to_table=clip_to_table)
        return [stroke[:, ::-1] for stroke in swapped]
    if axis != 0 or cell <= 0 or not 0 <= overlap < 1 or min(brush_size) <= 0 or (margin < 0).any():
        raise ValueError("Invalid brush/coverage settings")
    if not np.any(grid):
        return []
    origin = np.asarray(origin, float)
    brush = np.asarray(brush_size, float)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grid.astype(np.uint8))
    strokes = []
    for component in sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA]):
        yy, xx = np.nonzero(labels == component)
        low = origin + np.array([xx.min(), yy.min()]) * cell - margin
        high = origin + np.array([xx.max() + 1, yy.max() + 1]) * cell + margin
        height = high[1] - low[1]
        rows = max(1, int(np.ceil(max(0, height - brush[1]) / (brush[1] * (1 - overlap)))) + 1)
        centers = (np.array([(low[1] + high[1]) / 2]) if rows == 1 else
                   np.linspace(low[1] + brush[1] / 2, high[1] - brush[1] / 2, rows))
        for y in centers:
            a = np.array([low[0], y])
            b = np.array([high[0], y])
            bounds = np.array(grid.shape[::-1]) * cell
            if clip_to_table:
                safe_low=origin+edge_margin+brush/2
                safe_high=origin+bounds-edge_margin-brush/2
                if (safe_low>=safe_high).any():raise ValueError('Brush does not fit on the table')
                a=np.clip(a,safe_low,safe_high);b=np.clip(b,safe_low,safe_high)
            swept_low = np.minimum(a, b) - brush / 2 - origin
            swept_high = np.maximum(a, b) + brush / 2 - origin
            if (swept_low < edge_margin-1e-10).any() or (swept_high > bounds - edge_margin+1e-10).any():
                raise ValueError("Observed water requires wiping too close to a table edge")
            if len(strokes) % 2:
                a, b = b, a
            strokes.append(np.stack([a, b]))
    return strokes


class WipeCoveragePlanner:
    """Image-derived wet history and a finite verification pass after detection clears.

    Every update follows one newly captured photo. No fluid or cloth truth is
    accepted. Verification covers the entire photographed wet history,
    including gaps between disconnected traces and subpixel uncertain edges.
    """
    def __init__(self, brush_size, margin=(.009, .018), minimum_pull_m=.12, verification_pass=True,
                 clip_to_table=False,edge_margin=.025,batch_coverage=False):
        self.brush_size, self.margin = brush_size, margin
        self.minimum_pull_m = minimum_pull_m
        self.verification_pass = verification_pass
        self.clip_to_table,self.edge_margin=clip_to_table,edge_margin
        self.batch_coverage=batch_coverage
        self.history = self.fine_history = self.last_seen = None
        self.origin, self.cell = None, None
        self.queue = None
        self.verified_clear = False
        self.verification_actions = 0
        self.previous_grid = self.previous_stroke = None
        self.ineffective_strokes = []
        self.lateral_retry_m = 0.
        self.observed_progress = None

    @property
    def detection_history(self):
        return None if self.history is None else (self.history, self.origin, self.cell)

    @staticmethod
    def enclosing_region(grid):
        out = np.zeros_like(grid)
        y, x = np.nonzero(grid)
        if len(x):
            out[y.min():y.max() + 1, x.min():x.max() + 1] = 1
        return out

    def _strokes(self, grid, *, overlap=.3):
        strokes = coverage_strokes(grid, self.origin, self.cell, brush_size=self.brush_size,
                                   margin=self.margin, axis=1, overlap=overlap,
                                   clip_to_table=self.clip_to_table,edge_margin=self.edge_margin)
        # The validated physical grasp pulls the trailing wipe along +Y.
        strokes = [s[np.argsort(s[:, 1])] for s in strokes]
        # A short lowering-and-dragging motion leaves the free tail folded
        # forward. Allow enough travel to drape before crossing a tiny trace.
        # This uses tool geometry and the observed region, never liquid truth.
        for stroke in strokes:
            if np.ptp(stroke[:, 1]) < self.minimum_pull_m:
                center = stroke[:, 1].mean()
                stroke[:, 1] = center + np.array([-.5, .5]) * self.minimum_pull_m
            low = stroke.min(0) - np.asarray(self.brush_size) / 2
            high = stroke.max(0) + np.asarray(self.brush_size) / 2
            limit = self.origin + np.array(grid.shape[::-1]) * self.cell
            if self.clip_to_table:
                safe_low=self.origin+self.edge_margin+np.asarray(self.brush_size)/2
                safe_high=limit-self.edge_margin-np.asarray(self.brush_size)/2
                length=np.ptp(stroke[:,1])
                if length>safe_high[1]-safe_low[1]:raise ValueError('Draping pull does not fit on the table')
                shift=np.clip(stroke[:,1].mean(),safe_low[1]+length/2,safe_high[1]-length/2)-stroke[:,1].mean()
                stroke[:,1]+=shift
                stroke[:,0]=np.clip(stroke[:,0],safe_low[0],safe_high[0])
                low=stroke.min(0)-np.asarray(self.brush_size)/2
                high=stroke.max(0)+np.asarray(self.brush_size)/2
            if (low < self.origin + self.edge_margin-1e-10).any() or (high > limit - self.edge_margin+1e-10).any():
                raise ValueError('Draping pull requires wiping too close to a table edge')
        y, x = np.nonzero(grid)
        points = self.origin + (np.c_[x, y] + .5) * self.cell
        def score(stroke):
            lo = stroke.min(0) - np.asarray(self.brush_size) / 2
            hi = stroke.max(0) + np.asarray(self.brush_size) / 2
            return int(((points >= lo).all(1) & (points <= hi).all(1)).sum())
        return sorted(strokes, key=score, reverse=True)

    @staticmethod
    def _same_track(a, b):
        overlap = min(a[:, 1].max(), b[:, 1].max()) - max(a[:, 1].min(), b[:, 1].min())
        shorter = min(np.ptp(a[:, 1]), np.ptp(b[:, 1]))
        return abs(a[:, 0].mean() - b[:, 0].mean()) < .006 and overlap > .7 * shorter

    def _choose_effective_track(self, strokes, grid):
        if self.previous_grid is not None:
            previous = int(self.previous_grid.sum())
            self.observed_progress = (previous - int(grid.sum())) / max(previous, 1)
            union = int(((self.previous_grid > 0) | (grid > 0)).sum())
            iou = int(((self.previous_grid > 0) & (grid > 0)).sum()) / max(union, 1)
            if iou > .8 and self.observed_progress < .1 and self.previous_stroke is not None:
                self.ineffective_strokes.append(self.previous_stroke.copy())
            elif self.observed_progress > .15 or iou < .5:
                self.ineffective_strokes = []
        allowed = [s for s in strokes if not any(self._same_track(s, old) for old in self.ineffective_strokes)]
        self.lateral_retry_m = 0.
        if not allowed:
            # Contact width changes as the soft wipe drapes. Search nearby
            # parallel pulls when RGB proves that a nominal track was ineffective.
            for offset in [.010, -.010, .020, -.020]:
                trial = strokes[0] + [offset, 0.]
                low = trial.min(0) - np.asarray(self.brush_size) / 2
                high = trial.max(0) + np.asarray(self.brush_size) / 2
                limit = self.origin + np.array(grid.shape[::-1]) * self.cell
                if ((low >= self.origin + .025).all() and (high <= limit - .025).all()
                        and not any(self._same_track(trial, old) for old in self.ineffective_strokes)):
                    allowed = [trial]
                    self.lateral_retry_m = offset
                    break
        chosen = allowed[0] if allowed else strokes[0]
        self.previous_grid, self.previous_stroke = grid.copy(), chosen.copy()
        return [chosen, *[s for s in strokes if not np.array_equal(s, chosen)]]

    def update(self, grid, origin, cell, *, fine=False):
        if self.history is None:
            self.history = np.zeros_like(grid)
            self.origin, self.cell = np.asarray(origin).copy(), cell
        if (self.history.shape != grid.shape or not np.allclose(self.origin, origin)
                or self.cell != cell):
            raise ValueError('Water history requires the same metric desk grid')
        if np.any(grid):
            self.history |= grid
            self.last_seen = grid.copy()
            if fine:
                if self.fine_history is None:
                    self.fine_history = np.zeros_like(grid)
                self.fine_history |= grid
            self.queue, self.verified_clear = None, False
            planning_grid = self.enclosing_region(grid) if fine or self.batch_coverage else grid
            candidates = self._strokes(planning_grid)
            candidates = self._choose_effective_track(candidates, grid)
            return candidates, 'Fine trace cleanup' if fine else 'Observed water'
        self.previous_grid = self.previous_stroke = None
        self.observed_progress, self.lateral_retry_m = None, 0.
        if not self.verification_pass:
            self.queue=[]
            self.verified_clear=True
            return [], 'No visible water; awaiting agent stop decision'
        if self.queue is None:
            # Even when fine traces were visible, their footprint can exclude
            # subpixel remnants elsewhere in the earlier photographed puddle.
            # Cover the entire image-derived history before declaring it clear.
            region = self.history
            self.queue = [] if region is None else self._strokes(self.enclosing_region(region), overlap=.1)
        if self.queue:
            action = self.queue.pop(0)
            self.verification_actions += 1
            return [action, *self.queue], 'Verify observed trace neighborhood'
        self.verified_clear = True
        return [], 'No visible water after neighborhood verification'


def swept_coverage(grid, origin, cell, strokes, brush_size):
    yy, xx = np.nonzero(grid)
    points = np.asarray(origin) + (np.c_[xx, yy] + .5) * cell
    if not len(points):
        return 1.0
    covered = np.zeros(len(points), bool)
    for a, b in strokes:
        lo = np.minimum(a, b) - np.asarray(brush_size) / 2
        hi = np.maximum(a, b) + np.asarray(brush_size) / 2
        covered |= ((points >= lo) & (points <= hi)).all(1)
    return float(covered.mean()) if len(points) else 1.0
