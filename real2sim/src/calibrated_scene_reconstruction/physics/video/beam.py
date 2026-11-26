"""A slender body clamped at one end that bends under its own weight in a vertical plane: what its sag and its swing
measure (meta.json "motion": "beam"; perception by vidphys.soft).

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.beam real/<name> [...]         # physics.json
    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.beam --summary real/<name>     # summary.mp4
    tools/run_isaac.sh apps/video_soft.py real/<name> [--render]          # the same chain in PhysX, beside the video

Gravity is the only force a video shows. A beam held straight and let go sags by the ratio of its stiffness to its
weight, so the video gives the soft material's modulus over the density of whatever loads it (a specific modulus,
m^2/s^2); a modulus in pascals takes a density, which is a prior like a rigid body's mass. The swing on the way
there is quicker than that modulus allows wherever the material is viscoelastic (foam): it measures how much
stiffer the body is while it moves, and how fast that lets go.

The model is a planar chain: rigid lumps (the stiff parts, or slices of a uniform beam) joined by beam elements that
bend, shear and stretch (Timoshenko, corotational: the element's strains are measured against its own chord, so
rotations may be large; the material stays linear). A cube of foam between two blocks is one short element whose
shear gives as much as its bending. The material is a standard linear solid: a spring beside a spring in line with
a dashpot.

meta.json, beside vidphys.soft's fields:
  "parts": {"soft": prompt of the parts that deform (none: the whole body is of one soft material),
            "soft_density", "stiff_density": kg/m^3 of the two kinds, what they look like they are made of}
  "section": {"width_over_thickness": 1.0}   the section across the plane it bends in over the one in it

What the picture has to show: the body held straight, then let go, then at rest; the camera to one side of the
plane it swings in. The cut at the body's free end is not drawn (silhouette).
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document, write_document
import numpy as np

from calibrated_scene_reconstruction.core.geometry import mask_iou

LEAST_OVERLAP = 0.7  # of the fitted body with the mask (IoU), held and hanging: under it nothing is read

G = 9.81
AIR = 1.2  # kg/m^3
SHEAR_K = 5.0 / 6.0  # shear correction of a rectangular section


class Chain:
    """parts: [{"length" m, "soft": bool, "density" kg/m^3}] from the clamp outwards; a square section of side
    `side` unless `width` (across the plane) is given. Soft parts become `per_soft` elements each, with the lumps
    between them carrying the part's own mass; stiff parts are one lump each."""

    def __init__(self, parts, side, width=None, per_soft=1):
        width = side if width is None else width
        self.area, self.inertia_area = side * width, width * side ** 3 / 12.0
        lumps, elems, x, prev, pending = [], [], 0.0, -1, None  # lump: [x0, x1, mass, J]; elem: [a | -1, b, L0, x]
        for k, p in enumerate(parts):
            L, rho = float(p["length"]), float(p["density"])
            if not p["soft"]:
                m = rho * self.area * L
                lumps.append([x, x + L, m, m * (L ** 2 + side ** 2) / 12.0])
                prev = len(lumps) - 1
                if pending is not None:  # the element before ends on this lump, which takes half its mass
                    elems[pending[0]][1] = prev
                    lumps[prev][2] += pending[1]
                    pending = None
                x += L
                continue
            Le, me = L / per_soft, rho * self.area * L / per_soft  # an element's mass goes half to each of its ends
            for j in range(per_soft):
                if prev >= 0:
                    lumps[prev][2] += me / 2
                if j == per_soft - 1 and k < len(parts) - 1 and not parts[k + 1]["soft"]:
                    elems.append([prev, -2, Le, x])
                    pending = (len(elems) - 1, me / 2)
                else:
                    lumps.append([x + Le, x + Le, me / 2, -Le])  # a point: its inertia once its mass is known
                    elems.append([prev, len(lumps) - 1, Le, x])
                    prev = len(lumps) - 1
                x += Le
        for l in lumps:
            if l[3] < 0:
                l[3] = l[2] * (l[3] ** 2 + side ** 2) / 12.0
        self.lumps, self.elems = np.array(lumps), np.array(elems, float).reshape(-1, 4)
        self.length, self.side = x, side
        self.n = len(lumps)
        c = 0.5 * (self.lumps[:, 0] + self.lumps[:, 1])
        self.rest = np.concatenate([np.stack([c, np.zeros(self.n), np.zeros(self.n)], 1).ravel(), np.zeros(3 * self.n)])
        self.half = 0.5 * (self.lumps[:, 1] - self.lumps[:, 0])
        self.mass, self.J = self.lumps[:, 2], np.maximum(self.lumps[:, 3], 1e-12)
        self.a = self.elems[:, 0].astype(int)
        self.b = self.elems[:, 1].astype(int)
        self.L0, self.x0 = self.elems[:, 2], self.elems[:, 3]
        self.fixed = self.a < 0
        self.tilt = 0.0  # rad: how far above level the body is held; gravity leans by it in the chain's frame
        ne = len(self.L0)
        self.Aa, self.Ab = np.zeros((ne, self.n)), np.zeros((ne, self.n))  # element end -> lump
        self.Aa[np.nonzero(~self.fixed)[0], self.a[~self.fixed]] = 1.0
        self.Ab[np.arange(ne), self.b] = 1.0
        self.ha, self.hb = self.Aa @ self.half, self.Ab @ self.half

    def weight(self, g=G):
        """Gravity as generalised forces (3 n,)."""
        f = np.zeros(3 * self.n)
        f[0::3], f[1::3] = -self.mass * g * np.sin(self.tilt), -self.mass * g * np.cos(self.tilt)
        return f

    # Everything below takes states with any leading shape: (..., 3 n) for q and v, (..., E) per element.

    def _ends(self, q):
        x, z, ph = q[..., 0::3], q[..., 1::3], q[..., 2::3]
        pha, phb = ph @ self.Aa.T, ph @ self.Ab.T
        ax = np.where(self.fixed, self.x0, x @ self.Aa.T + self.ha * np.cos(pha))
        az = np.where(self.fixed, 0.0, z @ self.Aa.T + self.ha * np.sin(pha))
        bx = x @ self.Ab.T - self.hb * np.cos(phb)
        bz = z @ self.Ab.T - self.hb * np.sin(phb)
        return ax, az, pha, bx, bz, phb

    def nodes(self, q):
        """Ends of every element of one state: (p1 (E, 2), phi1 (E,), p2, phi2); the clamp is at the origin, level."""
        ax, az, pha, bx, bz, phb = self._ends(np.asarray(q, float))
        return np.stack([ax, az], -1), pha, np.stack([bx, bz], -1), phb

    def strains(self, q, v=None):
        """Each element's stretch and the turn of its two ends against its chord: e (..., E, 3), their rates when v
        is given, and the geometry `assemble` needs."""
        ax, az, pha, bx, bz, phb = self._ends(q)
        dx, dz = bx - ax, bz - az
        l = np.hypot(dx, dz)
        c, s = dx / l, dz / l
        beta = np.arctan2(dz, dx)
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
        e = np.stack([l - self.L0, wrap(pha - beta), wrap(phb - beta)], -1)
        geo = (c, s, l, pha, phb)
        if v is None:
            return e, None, geo
        vx, vz, w = v[..., 0::3], v[..., 1::3], v[..., 2::3]
        wa, wb = w @ self.Aa.T, w @ self.Ab.T
        vax = vx @ self.Aa.T - self.ha * np.sin(pha) * wa
        vaz = vz @ self.Aa.T + self.ha * np.cos(pha) * wa
        vbx = vx @ self.Ab.T + self.hb * np.sin(phb) * wb
        vbz = vz @ self.Ab.T - self.hb * np.cos(phb) * wb
        dvx, dvz = vbx - vax, vbz - vaz
        dbeta = (-s * dvx + c * dvz) / l
        de = np.stack([c * dvx + s * dvz, wa - dbeta, wb - dbeta], -1)
        return e, de, geo

    def stress(self, e, E, nu):
        """Axial force and the two end moments (..., E, 3) of strains e, for a modulus E (..., 1 or scalar)."""
        E = np.asarray(E, float)[..., None] if np.ndim(E) else E
        GA_E = SHEAR_K / (2 * (1 + nu)) * self.area  # GA / E
        phi = 12 * self.inertia_area / (GA_E * self.L0 ** 2)
        k = self.inertia_area / (self.L0 * (1 + phi))
        N = E * self.area / self.L0 * e[..., 0]
        M1 = E * k * ((4 + phi) * e[..., 1] + (2 - phi) * e[..., 2])
        M2 = E * k * ((2 - phi) * e[..., 1] + (4 + phi) * e[..., 2])
        return np.stack([N, M1, M2], -1)

    def assemble(self, sig, geo):
        """Generalised forces (..., 3 n) on the lumps from the elements' N, M1, M2."""
        c, s, l, pha, phb = geo
        N, M1, M2 = sig[..., 0], sig[..., 1], sig[..., 2]
        V = (M1 + M2) / l
        Fax, Faz = c * N + V * s, s * N - V * c  # what the element pulls its first end with; the second: the opposite
        rax, raz = self.ha * np.cos(pha), self.ha * np.sin(pha)
        rbx, rbz = -self.hb * np.cos(phb), -self.hb * np.sin(phb)
        fx = Fax @ self.Aa - Fax @ self.Ab
        fz = Faz @ self.Aa - Faz @ self.Ab
        fp = (-M1 + rax * Faz - raz * Fax) @ self.Aa + (-M2 - rbx * Faz + rbz * Fax) @ self.Ab
        return np.stack([fx, fz, fp], -1).reshape(fx.shape[:-1] + (3 * self.n,))

    def forces(self, q, v, E, nu, tau):
        """Generalised forces from the elements, gravity excluded; tau: a Kelvin-Voigt damping time."""
        e, de, geo = self.strains(q, v)
        return self.assemble(self.stress(e + tau * de, E, nu), geo)

    def step_limit(self, E, nu=0.3):
        """The largest explicit time step that is stable: from the highest frequency of the chain at rest."""
        n3 = 3 * self.n
        q0, h = self.rest[:n3], 1e-7
        Kmat = np.empty((n3, n3))
        f0 = self.forces(q0, np.zeros(n3), E, nu, 0.0)
        for i in range(n3):
            q = q0.copy()
            q[i] += h
            Kmat[:, i] = -(self.forces(q, np.zeros(n3), E, nu, 0.0) - f0) / h
        minv = 1.0 / np.stack([self.mass, self.mass, self.J], 1).ravel()
        w2 = np.abs(np.linalg.eigvals(minv[:, None] * Kmat)).max()
        return 1.0 / np.sqrt(w2)

    def swing(self, t, E, nu=0.3, E1=0.0, relax=1.0, tau=0.0, release=0.0, g=G, dt=None, mass_scale=1.0,
              carry=None, air=None):
        """States q (P, T, 3 n) at times t for P sets of parameters at once (each of E, E1, relax, tau, release,
        mass_scale a scalar or (P,)): held straight along its axis until `release`, then free. The material is a
        standard linear solid: a spring E beside a spring E1 in line with a dashpot, which lets go with the time
        `relax`; a quick swing feels E + E1, the body at rest E. tau adds a Kelvin-Voigt term. mass_scale: every
        lump that many times as heavy. Explicit (RK4) at a fixed step.

        carry: (times (N,), places (N,) m) of the clamp along the chain's z (across the body as it is held): the
        states are in the clamp's own frame, in which the body feels the clamp's acceleration; `release` is then
        one time for all. air: {"width" m across the plane, "cd" (scalar or (P,)), "added"}: the air a wide thin
        body moves through: a drag 1/2 rho cd |v_n| v_n per area on the speed across each lump, and the air that
        has to move with it, `added` x rho pi/4 width^2 per length, as mass against accelerations across it."""
        t = np.asarray(t, float)
        par = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, float))
                                    for x in (E, E1, relax, tau, release, mass_scale)])
        E, E1, relax, tau, release, ms = [x.astype(float) for x in par]
        P, n3, ne = len(E), 3 * self.n, len(self.L0)
        if dt is None:
            dt = min(1.4 * self.step_limit(float((E + E1).max()), nu) * np.sqrt(ms.min())
                     / (1 + float(tau.max()) / 0.002), 1e-3)
        span = float(t[-1] - release.min()) + 2 * dt
        steps = max(int(np.ceil(span / dt)), 1)
        keep = max(int(round(0.004 / dt)), 1)
        minv = (1.0 / np.stack([self.mass, self.mass, self.J], 1).ravel())[None] / ms[:, None]
        grav = self.weight(g)[None] * ms[:, None]
        tc = tau[:, None, None]
        c_vel = c_acc = None
        if carry is not None:
            from scipy.interpolate import CubicSpline

            cs = CubicSpline(np.asarray(carry[0], float), np.asarray(carry[1], float), bc_type="clamped")
            c_vel, c_acc = cs.derivative(1), cs.derivative(2)
        if air is not None:
            trib = 2 * self.half + 0.5 * (self.Aa.T @ self.L0 + self.Ab.T @ self.L0)  # the length each lump stands for
            area = air["width"] * trib
            cd = np.broadcast_to(np.atleast_1d(np.asarray(air["cd"], float)), (P,))[:, None]
            m_air = air.get("added", 0.0) * AIR * np.pi / 4 * air["width"] ** 2 * trib
            heavy = m_air / (self.mass[None] * ms[:, None] + m_air)  # the air's part of the mass across a lump

        def rhs(q, v, ev, now):
            e, de, geo = self.strains(q, v)
            sig = self.stress(e + tc * de, E, nu) + self.stress(e - ev, E1, nu)
            f = self.assemble(sig, geo) + grav
            if air is not None:
                nx, nz = -np.sin(q[:, 2::3]), np.cos(q[:, 2::3])  # across each lump
                vz = v[:, 1::3] + (float(c_vel(now)) if c_vel is not None else 0.0)
                vn = v[:, 0::3] * nx + vz * nz
                drag = -0.5 * AIR * cd * area * np.abs(vn) * vn
                f = f.copy()
                f[:, 0::3] += drag * nx
                f[:, 1::3] += drag * nz
            acc = f * minv
            if air is not None:
                an = heavy * (acc[:, 0::3] * nx + acc[:, 1::3] * nz)
                acc[:, 0::3] -= an * nx
                acc[:, 1::3] -= an * nz
            if c_acc is not None:
                acc[:, 1::3] -= float(c_acc(now))
            return v, acc, (e - ev) / relax[:, None, None]

        q = np.tile(self.rest[:n3], (P, 1))
        v, ev = np.zeros((P, n3)), np.zeros((P, ne, 3))
        store = [q.copy()]
        t0 = float(release.min())
        for k in range(steps):
            now = t0 + k * dt
            a1 = rhs(q, v, ev, now)
            a2 = rhs(q + 0.5 * dt * a1[0], v + 0.5 * dt * a1[1], ev + 0.5 * dt * a1[2], now + 0.5 * dt)
            a3 = rhs(q + 0.5 * dt * a2[0], v + 0.5 * dt * a2[1], ev + 0.5 * dt * a2[2], now + 0.5 * dt)
            a4 = rhs(q + dt * a3[0], v + dt * a3[1], ev + dt * a3[2], now + dt)
            q = q + dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
            v = v + dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
            ev = ev + dt / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
            if (k + 1) % keep == 0:
                store.append(q.copy())
        store = np.stack(store, 1)  # (P, S, 3 n) at multiples of keep * dt after each one's release
        ts = np.clip((t[None, :] - release[:, None]) / (keep * dt), 0.0, store.shape[1] - 1.0)
        i0 = np.floor(ts).astype(int).clip(0, store.shape[1] - 2)
        w = (ts - i0)[..., None]
        rows = np.arange(P)[:, None]
        return (1 - w) * store[rows, i0] + w * store[rows, i0 + 1]

    def rollout(self, t, E, nu=0.3, tau=0.0, release=0.0, g=G):
        """State q (T, 3 n) at times t of one elastic chain: held straight and level until `release`, then free.
        Implicit (LSODA): for chains too stiff for `swing`."""
        from scipy.integrate import solve_ivp

        n3 = 3 * self.n
        minv = 1.0 / np.stack([self.mass, self.mass, self.J], 1).ravel()
        grav = self.weight(g)

        def rhs(_t, y):
            return np.concatenate([y[n3:], (self.forces(y[:n3], y[n3:], E, nu, tau) + grav) * minv])

        t = np.asarray(t, float)
        out = np.tile(self.rest[:n3], (len(t), 1))
        after = t > release
        if after.any():
            sol = solve_ivp(rhs, (0.0, float(t[after][-1] - release) + 1e-9), self.rest, t_eval=t[after] - release,
                            method="LSODA", rtol=1e-6, atol=1e-8)
            out[after] = sol.y[:n3].T[: after.sum()]
        return out

    def axis(self, q, s):
        """Points (len(s), 2) of the chain's axis at rest abscissae s (m from the clamp): on the lumps where they are
        rigid, on a cubic through the element's ends in between."""
        q = np.asarray(q)
        x, z, ph = q[0::3], q[1::3], q[2::3]
        s = np.asarray(s, float)
        out = np.full((len(s), 2), np.nan)
        for i in range(self.n):
            x0, x1 = self.lumps[i, 0], self.lumps[i, 1]
            m = (s >= x0) & (s <= x1) if x1 > x0 else np.zeros(len(s), bool)
            r = s[m] - 0.5 * (x0 + x1)
            out[m] = np.stack([x[i] + r * np.cos(ph[i]), z[i] + r * np.sin(ph[i])], 1)
        pa, pha, pb, phb = self.nodes(q)
        for e in range(len(self.L0)):
            m = (s >= self.x0[e]) & (s <= self.x0[e] + self.L0[e]) & np.isnan(out[:, 0])
            if not m.any():
                continue
            w = (s[m] - self.x0[e]) / self.L0[e]
            L = self.L0[e]
            ta, tb = np.array([np.cos(pha[e]), np.sin(pha[e])]) * L, np.array([np.cos(phb[e]), np.sin(phb[e])]) * L
            h00, h10 = 2 * w ** 3 - 3 * w ** 2 + 1, w ** 3 - 2 * w ** 2 + w
            h01, h11 = -2 * w ** 3 + 3 * w ** 2, w ** 3 - w ** 2
            out[m] = h00[:, None] * pa[e] + h10[:, None] * ta + h01[:, None] * pb[e] + h11[:, None] * tb
        return out


# --- the video: the plane the beam moves in, its silhouette, the fit -----------------------------------------------


class View:
    """The vertical plane a beam bends in, as the camera sees it. In-plane coordinates are the chain's: s along the
    body as it is held (from the clamp towards the tip; level unless `tilt`, radians above level) and z across it,
    upwards; w is across the plane, towards the camera."""

    def __init__(self, K, T_world_cam, origin, along, across, tilt=0.0):
        self.K, self.T = np.asarray(K, float), np.asarray(T_world_cam, float)
        self.o, self.a, self.n = np.asarray(origin, float), np.asarray(along, float), np.asarray(across, float)
        self.tilt = float(tilt)

    def world(self, s, z, w=0.0):
        c, sn = np.cos(self.tilt), np.sin(self.tilt)
        s, z = np.asarray(s, float), np.asarray(z, float)
        return (self.o + np.multiply.outer(s * c - z * sn, self.a) + np.multiply.outer(s * sn + z * c, [0, 0, 1.0])
                + np.multiply.outer(w, self.n))

    def project(self, X):
        Xc = (X - self.T[:3, 3]) @ self.T[:3, :3]
        return np.stack([self.K[0, 0] * Xc[..., 0] / Xc[..., 2] + self.K[0, 2],
                         self.K[1, 1] * Xc[..., 1] / Xc[..., 2] + self.K[1, 2]], -1)

    def pixels(self, s, z, w=0.0):
        return self.project(self.world(s, z, w))

    def plane(self, uv):
        """(s, z) of the points of the plane that pixels (..., 2) show."""
        uv = np.asarray(uv, float)
        d = np.stack([(uv[..., 0] - self.K[0, 2]) / self.K[0, 0], (uv[..., 1] - self.K[1, 2]) / self.K[1, 1],
                      np.ones(uv.shape[:-1])], -1) @ self.T[:3, :3].T
        c = self.T[:3, 3]
        lam = ((self.o - c) @ self.n) / (d @ self.n)
        X = c + lam[..., None] * d - self.o
        h, v = X @ self.a, X[..., 2]
        ct, st = np.cos(self.tilt), np.sin(self.tilt)
        return h * ct + v * st, -h * st + v * ct

    def down(self, uv):
        """The unit direction (2,) in which things fall in the picture at pixel uv."""
        s, z = self.plane(uv)
        X = self.world(s, z)
        d = self.project(X - np.array([0, 0, 0.01])) - self.project(X)
        return d / np.linalg.norm(d)

    def moved(self, ds=0.0, dz=0.0, dw=0.0, flip=False):
        o = self.world(ds, dz, dw)
        return View(self.K, self.T, o, -self.a if flip else self.a, self.n, -self.tilt if flip else self.tilt)

    def turned(self, deg):
        """The plane turned about the vertical through its origin."""
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return View(self.K, self.T, self.o, R @ self.a, R @ self.n, self.tilt)


LEVEL = 5.0  # deg: a body held within this of level is taken to be level (MoGe-2's depth along it is no better)


def view_of(per):
    """The plane through MoGe-2's points of the body as it is held: their principal direction and gravity. Returns
    (View, the direction's tilt in degrees as measured)."""
    from .scene import camera_pose

    n = np.median(per["plane_n"], 0)
    T = camera_pose(n / np.linalg.norm(n), float(np.median(per["plane_h"])))
    P = per["object_points"] @ T[:3, :3].T + T[:3, 3]
    c = np.median(P, 0)
    d = P - c
    d = d[np.linalg.norm(d, axis=1) < 4 * np.median(np.linalg.norm(d, axis=1))]
    ax = np.linalg.eigh(np.cov(d.T))[1][:, -1]
    a = np.array([ax[0], ax[1], 0.0])
    a /= np.linalg.norm(a)
    tilt = float(np.degrees(np.arcsin(np.clip(ax[2] * np.sign(ax @ a), -1, 1))))
    nrm = np.cross([0, 0, 1.0], a)
    if (T[:3, 3] - c) @ nrm < 0:
        nrm = -nrm
    return View(per["K"], T, c, a, nrm, np.radians(tilt) if abs(tilt) > LEVEL else 0.0), tilt


def silhouette(chain, q, view, shape):
    """The chain as the camera sees it: (H, W) bool. Of its long faces the ones turned to the camera are drawn; its
    two ends are not: the cut at the clamp is against the clamp, and the one at the tip, small and seen at a slant,
    is the first thing a segmenter leaves out."""
    import cv2

    canvas = np.zeros(shape, np.uint8)
    h, w = 0.5 * chain.side, 0.5 * chain.area / chain.side
    x, z, ph = q[0::3], q[1::3], q[2::3]
    cam = view.T[:3, 3]
    sg, ws = np.array([1.0, -1.0, -1.0, 1.0]), np.array([w, w, -w, -w])

    def section(p, phi):  # the four corners of a cross-section at p, normal to the axis: world points
        up = np.array([-np.sin(phi), np.cos(phi)])
        return view.world(p[0] + h * sg * up[0], p[1] + h * sg * up[1], ws)

    def faces(A, B):
        mid = 0.25 * (A.sum(0) + B.sum(0))
        for i in range(4):
            j = (i + 1) % 4
            quad = np.array([A[i], A[j], B[j], B[i]])
            c = quad.mean(0)
            if (c - mid) @ (cam - c) <= 0:  # turned away
                continue
            uv = view.project(quad)
            if np.all(np.isfinite(uv)):
                cv2.fillPoly(canvas, [np.round(uv).astype(np.int32)], 1)

    pa, pha, pb, phb = chain.nodes(q)
    for e in range(len(chain.L0)):
        faces(section(pa[e], pha[e]), section(pb[e], phb[e]))
    for i in range(chain.n):
        if chain.half[i] > 0:
            t = np.array([np.cos(ph[i]), np.sin(ph[i])]) * chain.half[i]
            c = np.array([x[i], z[i]])
            faces(section(c - t, ph[i]), section(c + t, ph[i]))
    return canvas.astype(bool)


def iou(a, b):
    return mask_iou(a, b)


def release_frame(obj, view, least=0.03):
    """The first frame the body has begun to fall for good: its silhouette's centroid lower than where it was held by
    `least` of the body's length in the picture, in that frame and the five after (those of them that show it: a
    body that falls fast is a streak the segmenter may miss, and the frames missed just before count as fallen).
    A gripper that opens changes the mask too, but not downwards. Returns (frame | None, centroids (F, 2) px)."""
    cen = np.array([np.argwhere(o).mean(0)[::-1] if o.any() else [np.nan, np.nan] for o in obj])
    seen = np.isfinite(cen[:, 0])
    if seen.sum() < 12:
        return None, cen
    first = int(np.argmax(seen))
    down = view.down(cen[first])
    vs, us = np.nonzero(obj[first])
    size = float(np.hypot(np.ptp(us), np.ptp(vs)))
    for k in range(first + 3, len(obj) - 6):
        if not seen[first:k].sum() >= 3:
            continue
        fell = (cen[k:k + 6] - np.nanmedian(cen[first:k], 0)) @ down > least * size
        if seen[k:k + 6].sum() >= 3 and fell[seen[k:k + 6]].all():
            while k - 1 > first and not seen[k - 1]:
                k -= 1
            return k, cen
    return None, cen


def majority(masks):
    return masks.mean(0) > 0.5


def rest_shape(view, rest, hang, width_ratio=1.0):
    """Where the beam is held and how long and thick it is, from its silhouette at rest: the box whose picture
    covers the mask best. Returns (view with its origin at the clamp on the axis and s towards the tip, length,
    side)."""
    from scipy.optimize import minimize

    vs, us = np.nonzero(rest)
    s, z = view.plane(np.stack([us, vs], 1))
    lo, hi = np.percentile(s, [1, 99])
    # the clamp's end is the one the hanging body still covers
    vh, uh = np.nonzero(hang & rest)
    if not len(vh):
        raise ValueError("the body hangs nowhere near where it was held: no end of it is clamped")
    sh, _zh = view.plane(np.stack([uh, vh], 1))
    flip = abs(np.median(sh) - hi) < abs(np.median(sh) - lo)
    v = view.moved(ds=hi if flip else lo, dz=float(np.median(z)), flip=flip)
    L0 = hi - lo
    a0 = float(np.percentile(z, 95) - np.percentile(z, 5)) * 0.8

    def box(p):
        ds, dz, L, a = p
        ch = Chain([{"length": max(L, 1e-3), "soft": False, "density": 1.0}], max(a, 1e-3), max(a, 1e-3) * width_ratio)
        return ch, v.moved(ds=ds, dz=dz)

    def cost(p):
        ch, vv = box(p)
        return 1.0 - iou(silhouette(ch, ch.rest[:3], vv, rest.shape), rest)

    best = minimize(cost, [0.0, 0.0, L0, a0], method="Nelder-Mead",
                    options={"initial_simplex": np.array([[0.0, 0.0, L0, a0]] * 5) + np.vstack(
                        [np.zeros(4), np.diag([0.1 * L0, 0.3 * a0, 0.1 * L0, 0.3 * a0])]),
                             "xatol": 2e-4, "fatol": 1e-4, "maxiter": 400})
    ds, dz, L, a = best.x
    return v.moved(ds=ds, dz=dz), float(L), float(a), 1.0 - float(best.fun)


def along_face(view, mask, length, side, n=400):
    """Where along a straight, level body a mask lies: read on the middle line of the long face the camera sees
    best. (A mask's extent in the picture will not do: the top of a slice seen from one side is shifted along the
    body.) Returns (s (n,), on (n,) bool)."""
    h = 0.5 * side
    mid = view.world(0.5 * length, 0.0)
    z, w = max([(h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)],
               key=lambda f: (view.world(0.5 * length, f[0], f[1]) - mid) @ (view.T[:3, 3] - mid))
    s = np.linspace(0.0, length, n)
    uv = np.round(view.pixels(s, np.full(n, z), np.full(n, w))).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
    on = np.zeros(n, bool)
    on[ok] = mask[uv[ok, 1], uv[ok, 0]]
    return s, on


def parts_of(view, soft, length, side, meta):
    """The body's parts from the clamp outwards: where the soft parts' masks lie along the axis, stiff in between.
    Without masks the whole body is one soft part. Returns (parts, lead): a stiff part before the first soft one
    is the clamp's, `lead` m of it."""
    P = meta.get("parts", {})
    rho_s, rho_k = float(P.get("soft_density", 500.0)), float(P.get("stiff_density", 1000.0))
    iv = []
    for m in soft:
        s, on = along_face(view, m, length, side)
        if on.sum() >= 4:
            iv.append((float(s[on][0]), float(s[on][-1])))
    if not iv:
        return [{"length": length, "soft": True, "density": rho_s}], 0.0
    iv.sort(key=lambda r: r[0])
    out, x = [], 0.0
    for a, b in iv:
        a = max(a, x)
        if a > x:
            out.append({"length": float(a - x), "soft": False, "density": rho_k})
        if b > a:
            out.append({"length": float(b - a), "soft": True, "density": rho_s})
        x = max(x, b)
    if length - x > 0.15 * side:
        out.append({"length": float(length - x), "soft": False, "density": rho_k})
    else:
        out[-1]["length"] += float(length - x)
    lead = 0.0
    if not out[0]["soft"] and len(out) > 1:
        lead = out.pop(0)["length"]
    return out, lead


def hanging(chain, Es, nu=0.3, start=None):
    """The shapes (len(Es), 3 n) the chain hangs in for stiffnesses Es (best descending): each found from the one
    before, the first from `start` (default: straight)."""
    from scipy.optimize import root

    n3 = 3 * chain.n
    grav, zero = chain.weight(), np.zeros(n3)
    scale = np.stack([chain.mass, chain.mass, chain.mass * chain.length], 1).ravel() * G
    q, out = (chain.rest[:n3] if start is None else np.asarray(start, float)).copy(), []
    for E in Es:
        balance = lambda y: (chain.forces(y, zero, E, nu, 0.0) + grav) / scale
        sol = root(balance, q, method="hybr", tol=1e-10)
        if not sol.success or np.abs(sol.fun).max() > 1e-6:  # let it fall into place first
            fell = chain.rollout(np.array([0.0, 20.0]), E, nu, 0.05)[-1]
            sol = root(balance, fell, method="hybr", tol=1e-10)
            q = sol.x if sol.success else fell
        else:
            q = sol.x
        out.append(q.copy())
    return np.array(out)


def fit_static(chain, view, hang, nu=0.3, n=40):
    """The stiffness whose hanging shape covers the mask of the body at rest in the end: a ladder of n over three
    decades, then a finer one about its best rung. Returns (E, IoU, the shape q, the ladder [(E, IoU)])."""
    # from stiff enough to sag by about a hundredth of its length down to a thousandfold softer
    load = (chain.mass * 0.5 * (chain.lumps[:, 0] + chain.lumps[:, 1])).sum() * G  # moment at the clamp, N m
    Es = load * chain.length / (chain.inertia_area * 0.01) / 3.0 * np.logspace(0, -3, n)
    shapes = hanging(chain, Es, nu)
    score = np.array([iou(silhouette(chain, q, view, hang.shape), hang) for q in shapes])
    k = int(np.argmax(score))
    fine = np.geomspace(Es[max(k - 1, 0)], Es[min(k + 1, n - 1)], 13)
    shapes_f = hanging(chain, fine, nu, shapes[max(k - 1, 0)])
    score_f = np.array([iou(silhouette(chain, q, view, hang.shape), hang) for q in shapes_f])
    j = int(np.argmax(score_f))
    if score_f[j] > score[k]:
        return float(fine[j]), float(score_f[j]), shapes_f[j], list(zip(Es.tolist(), score.tolist()))
    return float(Es[k]), float(score[k]), shapes[k], list(zip(Es.tolist(), score.tolist()))


def period_of(t, y):
    """The period of a decaying swing: the lag of the autocorrelation's first peak after its first trough."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 20:
        return np.nan
    y = np.interp(t, t[ok], y[ok])
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    dt = float(np.median(np.diff(t)))
    up = 8
    tt = np.arange(t[0], t[-1], dt / up)
    yy = np.interp(tt, t, y)
    ac = np.correlate(yy, yy, "full")[len(yy) - 1:]
    ac /= ac[0]
    neg = np.nonzero(ac < 0)[0]
    if not len(neg):
        return np.nan
    k0 = neg[0]
    pos = np.nonzero(ac[k0:] > 0)[0]
    if not len(pos):
        return np.nan
    k1 = k0 + pos[0]
    nxt = np.nonzero(ac[k1:] < 0)[0]
    k2 = k1 + (nxt[0] if len(nxt) else len(ac) - k1)
    return float((k1 + int(np.argmax(ac[k1:k2]))) * dt / up)


def middle(chain, q, view):
    """The pixel (..., 2) at the middle of the chain's picture, cheaply: its lumps' and elements' centres seen by
    the camera, weighted by their lengths. It moves as the silhouette's centroid does."""
    ax, az, _pha, bx, bz, _phb = chain._ends(q)
    lump = 2 * chain.half
    s = np.concatenate([q[..., 0::3], 0.5 * (ax + bx)], -1)
    z = np.concatenate([q[..., 1::3], 0.5 * (az + bz)], -1)
    w = np.concatenate([lump, chain.L0])
    uv = view.pixels(s, z)
    return (uv * w[:, None]).sum(-2) / w.sum()


def last_frames(t):
    """How many frames the last two thirds of a second are (20 at 30 a second; never under 20, nor over a third of
    the video): the stretch over which a body that has come to rest is read."""
    return int(min(max(20, round(0.67 / float(np.median(np.diff(t))))), max(len(t) // 3, 1)))


def fit_swing(chain, view, t, cen, E, release0, nu=0.3, samples=64, rounds=7, seed=0):
    """How much stiffer the body is while it swings than when it hangs, how fast that lets go, and the moment of
    release: from the path of the silhouette's centroid, the hanging stiffness E held. Searched by sampling (the
    cross-entropy method), every round one batch of swings. Returns (E1, relax s, release s, the model's centroid
    path (F, 2) px, rms px)."""
    rng = np.random.default_rng(seed)
    use = np.isfinite(cen[:, 0]) & (t > release0 - 0.3)
    n_end = last_frames(t)
    end = np.nanmedian(cen[-n_end:], 0)

    def paths(X):
        q = chain.swing(t, E, nu, E * np.exp(X[:, 0]), np.exp(X[:, 1]), 0.0, X[:, 2])
        m = middle(chain, q, view)
        return m - m[:, -n_end:].mean(1, keepdims=True) + end

    mu, sd, best = np.array([0.0, np.log(0.15), release0]), np.array([1.2, 1.2, 0.05]), None
    for _ in range(rounds):
        X = mu + sd * rng.standard_normal((samples, 3))
        X[0] = mu
        loss = np.sqrt(np.mean(np.sum((paths(X) - cen)[:, use] ** 2, -1), 1))
        loss = np.nan_to_num(loss, nan=1e6)
        o = np.argsort(loss)
        mu, sd = X[o[:samples // 6]].mean(0), X[o[:samples // 6]].std(0) + 1e-3
        if best is None or loss[o[0]] < best[0]:
            best = (float(loss[o[0]]), X[o[0]].copy())
    rms, x = best
    return float(E * np.exp(x[0])), float(np.exp(x[1])), float(x[2]), paths(x[None])[0], rms


def lengthened(parts, more, meta):
    """The parts with `more` m added at the tip: a body held against a background of its own colour is not seen to
    its end, and what is not seen is of the kind that hides (the last part's, if that is stiff)."""
    parts = [dict(p) for p in parts]
    if more <= 0:
        return parts
    if not parts[-1]["soft"] or "parts" not in meta:
        parts[-1]["length"] += more
    else:
        parts.append({"length": more, "soft": False, "density": float(meta["parts"].get("stiff_density", 1000.0))})
    return parts


def elements_for(parts, side, most=10):
    """How many elements each soft part is cut into: about one per thickness of the longest one."""
    soft = [p["length"] for p in parts if p["soft"]]
    return int(np.clip(round(max(soft) / side), 1, most)) if soft else 1


def chain_of(b):
    """The chain a physics.json "beam" describes."""
    chain = Chain(b["parts"], b["side"], b.get("width"), b.get("elements_per_soft", 1))
    chain.tilt = b.get("tilt", 0.0)
    return chain


def measure(per, meta, search=(64, 7)):
    """physics.json of a recording whose motion is "beam", from its perception (perception.npz's fields).
    search: (samples, rounds) of the swing's fit."""
    obj, t = per["object"], per["t"]
    out = {"motion": "beam", "fps": float(1.0 / np.median(np.diff(t)))}
    if str(per["camera"]) == "moving":
        return dict(out, measured=False, reason="the camera travels")
    view, tilt = view_of(per)
    rel, cen = release_frame(obj, view)
    if rel is None or rel < 4:
        return dict(out, measured=False, reason="the body is never seen held and then let go")
    n_end = last_frames(t)
    still = np.linalg.norm(cen[-n_end:] - np.nanmedian(cen[-n_end:], 0), axis=1)
    # (a mask's edge flickers: three pixels of a picture 960 wide, or a twentieth of the body's size)
    if np.nanmax(still) > max(3.0 * obj.shape[2] / 960, 0.05 * np.sqrt(obj[-n_end:].reshape(n_end, -1).sum(1).mean())):
        return dict(out, measured=False, reason="the body is still swinging when the video ends")
    if meta.get("parts", {}).get("soft") and not len(per["soft_parts"]):
        return dict(out, measured=False, reason="the soft parts meta.json names are not found on the body at rest")
    rest, hang = majority(obj[:max(rel - 3, 1)]), majority(obj[-n_end:])
    if not rest.any() or not hang.any():
        return dict(out, measured=False, reason="the body is not seen while it is held, or not at rest in the end")
    width_ratio = float(meta.get("section", {}).get("width_over_thickness", 1.0))
    v, L, a, i_rest = rest_shape(view, rest, hang, width_ratio)
    parts0, lead = parts_of(v, per["soft_parts"], L, a, meta)
    v = v.moved(ds=lead)
    per_soft = elements_for(parts0, a)

    def chain_with(more):
        chain = Chain(lengthened(parts0, more, meta), a, a * width_ratio, per_soft)
        chain.tilt = v.tilt
        return chain

    best = None
    for more in np.arange(0.0, 0.5 * L, 0.2 * a):  # the tip is read off the hanging body, seen whole
        chain = chain_with(more)
        E, sc, qh, _ladder = fit_static(chain, v, hang)
        if best is None or sc > best[1]:
            best = (more, sc, E, qh, chain)
        elif sc < best[1] - 0.05:
            break
    more, i_hang, E, qh, chain = best
    parts = lengthened(parts0, more, meta)
    P = meta.get("parts", {})
    loaded = any(not p["soft"] for p in parts)  # stiff parts are the weight; without them the soft material's own
    rho = float(P.get("stiff_density", 1000.0) if loaded else P.get("soft_density", 500.0))
    given = ("stiff_density" if loaded else "soft_density") in P
    if min(i_rest, i_hang) < LEAST_OVERLAP:  # the fitted body is not the one in the picture: its sag says nothing
        return dict(out, measured=False,
                    reason=f"the fitted body covers the mask only {min(i_rest, i_hang):.2f} (held {i_rest:.2f}, "
                           f"hanging {i_hang:.2f}): the body's plane, its parts or its mask are not what the model "
                           "takes")
    tip = chain.axis(qh, [chain.length])[0]
    b = {"length": chain.length, "side": a, "width": a * width_ratio, "elements_per_soft": per_soft,
         "clamp_lead": lead, "unseen_tip": float(more), "parts": parts,
         "tilt_at_rest_deg": tilt, "tilt": v.tilt,
         # what hangs: the modulus at rest, over the density of what loads it (the prior: a modulus alone needs it)
         "modulus": E, "load_density": rho, "specific_modulus": E / rho, "poisson_assumed": 0.3,
         "load_density_from": "meta.json (a prior: what the loading part looks to be made of)" if given else
         "a default: no density was given, the modulus in pascals is a placeholder and specific_modulus the reading",
         "sag": float(-tip[1]), "sag_over_length": float(-tip[1] / chain.length),
         "angles_deg": np.degrees(qh[2::3]).tolist(), "iou_rest": i_rest, "iou_hanging": i_hang,
         "release_frame": int(rel), "release": float(t[rel]),
         "plane": {"origin": v.o.tolist(), "along": v.a.tolist(), "across": v.n.tolist()}}
    out.update(measured=True, beam=b, K=np.asarray(per["K"]).tolist(), T_world_cam=v.T.tolist())
    # what swings: stiffer by modulus_fast, which lets go in relax_time
    steps = (t[-1] - t[rel]) / (1.4 * chain.step_limit(3 * E))
    if steps > 2e5:
        b.update(modulus_fast=None, stiffening=None, relax_time=None,
                 swing="not fitted: the chain is too stiff for the explicit swing (thin, or of many elements)")
        return out
    E1, relax, t_rel, pred, rms = fit_swing(chain, v, t, cen, E, float(t[rel]), samples=search[0], rounds=search[1])
    qs = chain.swing(t, E, 0.3, E1, relax, 0.0, t_rel)[0]
    first = np.arange(rel, min(rel + int(round(1.5 * out["fps"])), len(t)))  # the swing's first second and a half
    i_swing = float(np.mean([iou(silhouette(chain, qs[k], v, obj[k].shape), obj[k]) for k in first]))
    after = t > t_rel + 0.1
    b.update(modulus_fast=E + E1, stiffening=(E + E1) / E, relax_time=relax, release=t_rel, iou_swing=i_swing,
             centroid_rms_px=rms, period=period_of(t[after], cen[after, 1]),
             period_model=period_of(t[after], pred[after, 1]))
    return out


def fit(path):
    """physics.json of a recording directory."""
    import json
    from pathlib import Path

    path = Path(path)
    return measure(np.load(path / "perception.npz"), read_document(path / "meta.json"))


def summary(path, seconds=None):
    """summary.mp4 of a measured beam: the video with its mask (green) and the fitted chain's outline (yellow) and
    axis (soft parts blue, stiff ones orange); beside it the chain from the side in metres, the height of the
    silhouette's centroid along time (video grey, chain yellow), and the numbers with where each comes from."""
    import json
    import subprocess
    from pathlib import Path

    import cv2

    from .scene import read_frames

    path = Path(path)
    phys = read_document(path / "physics.json")
    if not phys.get("measured"):
        return None
    b = phys["beam"]
    per = np.load(path / "perception.npz")
    obj, t = per["object"], per["t"]
    frames, fps = read_frames(path)
    if not b.get("modulus_fast"):
        return None
    chain = chain_of(b)
    pl = b["plane"]
    view = View(phys["K"], phys["T_world_cam"], pl["origin"], pl["along"], pl["across"], b.get("tilt", 0.0))
    qs = chain.swing(t, b["modulus"], b["poisson_assumed"], b["modulus_fast"] - b["modulus"], b["relax_time"], 0.0,
                     b["release"])[0]
    cen = np.array([np.argwhere(o).mean(0)[::-1] if o.any() else [np.nan, np.nan] for o in obj])
    mid = np.array([np.argwhere(m).mean(0)[::-1] if m.any() else [np.nan, np.nan]
                    for m in (silhouette(chain, q, view, obj[0].shape) for q in qs)])
    H, W = frames.shape[1:3]
    PW = 520
    last = len(t) if seconds is None else min(len(t), int(np.searchsorted(t, b["release"] + seconds)) + 1)
    first = max(int(b["release_frame"]) - 20, 0)
    tmp = path / "summary_raw.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + PW, H))
    L = chain.length
    side = (W + 20, 34, W + PW - 20, 34 + 200)  # the plane: s from -0.3 L to 1.3 L, z from 0.12 L down to -1.1 L
    plot = (W + 20, side[3] + 34, W + PW - 20, side[3] + 34 + 80)
    sx = (side[3] - side[1]) / (1.22 * L)
    to_side = lambda p: (int(0.5 * (side[0] + side[2]) + (p[0] - 0.5 * L) * sx),
                         int(side[1] + 0.12 * L * sx - p[1] * sx))
    ylo, yhi = np.nanmin(cen[first:last, 1]) - 5, np.nanmax(cen[first:last, 1]) + 5
    to_plot = lambda tt, y: (int(plot[0] + (tt - t[first]) / max(t[last - 1] - t[first], 1e-6) * (plot[2] - plot[0])),
                             int(plot[1] + (y - ylo) / (yhi - ylo) * (plot[3] - plot[1])))
    text = [f"length {L * 100:.1f} cm, side {b['side'] * 100:.1f} cm (the picture and MoGe-2's depth)",
            "parts from the clamp: " + " ".join(f"{'soft' if p['soft'] else 'stiff'} {p['length'] * 100:.1f}"
                                                for p in b["parts"]),
            f"hanging: sag {b['sag'] * 100:.1f} cm -> modulus {b['modulus'] / 1e3:.1f} kPa",
            f"   for a load of {b['load_density']:.0f} kg/m3 (a prior): E/rho {b['specific_modulus']:.1f} m2/s2",
            f"swinging: period {b['period']:.2f} s -> {b['stiffening']:.2f} x stiffer "
            f"({b['modulus_fast'] / 1e3:.0f} kPa),",
            f"   letting go in {b['relax_time'] * 1e3:.0f} ms",
            f"overlap of chain and mask: hanging {b['iou_hanging']:.2f}, swinging {b['iou_swing']:.2f}"]
    for k in range(first, last):
        img = np.zeros((H, W + PW, 3), np.uint8)
        im = frames[k][:, :, ::-1].copy()
        cs, _ = cv2.findContours(obj[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (0, 220, 0), 1)
        cs, _ = cv2.findContours(silhouette(chain, qs[k], view, obj[k].shape).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (0, 230, 255), 1)
        img[:, :W] = im
        cv2.putText(img, "green: the video's mask   yellow: the fitted chain", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # the chain from the side
        cv2.rectangle(img, side[:2], side[2:], (70, 70, 70), 1)
        cv2.putText(img, f"the beam's plane (m), t = {t[k] - b['release']:+.2f} s from release",
                    (side[0] + 4, side[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        x = 0.0
        for p in b["parts"]:
            ss = np.linspace(x, x + p["length"], 8)
            pts = np.array([to_side(q) for q in chain.axis(qs[k], ss)], np.int32)
            cv2.polylines(img, [pts], False, (255, 160, 60) if p["soft"] else (60, 160, 255),
                          max(int(b["side"] * sx * 0.45), 2),
                          cv2.LINE_AA)
            x += p["length"]
        cv2.line(img, to_side((0, -0.06 * L)), to_side((0, 0.06 * L)), (255, 255, 255), 2)
        cv2.putText(img, "blue: soft   orange: stiff   white: the clamp", (side[0] + 6, side[3] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        # the centroid's height along time
        cv2.rectangle(img, plot[:2], plot[2:], (70, 70, 70), 1)
        cv2.putText(img, "height of the silhouette's middle (px): video grey, chain yellow", (plot[0] + 4, plot[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        for y, col in ((cen[:, 1], (170, 170, 170)), (mid[:, 1], (0, 230, 255))):
            pts = np.array([to_plot(t[j], y[j]) for j in range(first, last) if np.isfinite(y[j])], np.int32)
            cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
        cv2.line(img, (to_plot(t[k], ylo)[0], plot[1]), (to_plot(t[k], ylo)[0], plot[3]), (255, 255, 255), 1)
        for j, line in enumerate(text):
            cv2.putText(img, line, (W + 16, plot[3] + 28 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1,
                        cv2.LINE_AA)
        vw.write(img)
    vw.release()
    out = path / "summary.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "22",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        tmp.unlink()
    except (OSError, subprocess.CalledProcessError):
        tmp.rename(out)
    return out


def main():
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="+")
    ap.add_argument("--summary", action="store_true", help="summary.mp4 of recordings already fitted")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    for p in args.recording:
        if args.summary:
            print(summary(p), flush=True)
            continue
        if args.skip_done and (Path(p) / "physics.json").exists():
            continue
        if not (Path(p) / "perception.npz").exists():
            continue
        try:
            from .scene import untimed

            why = untimed(read_document(Path(p) / "meta.json"))
            o = fit(p) if why is None else {"motion": "beam", "measured": False, "reason": why}
        except Exception as e:  # one recording's failure is its own
            o = {"motion": "beam", "measured": False, "reason": f"{type(e).__name__}: {e}"}
        write_document(Path(p) / "physics.json", o, indent=1)
        if o.get("measured") and not o["beam"].get("modulus_fast"):
            print(f"{p}: modulus {o['beam']['modulus'] / 1e3:.1f} kPa; swing {o['beam']['swing']}", flush=True)
        elif o.get("measured"):
            b = o["beam"]
            print(f"{p}: modulus {b['modulus'] / 1e3:.1f} kPa at {b['load_density']:.0f} kg/m^3 (x "
                  f"{b['stiffening']:.2f} swinging, for {b['relax_time'] * 1e3:.0f} ms), sag {b['sag'] * 1e3:.0f} mm "
                  f"of {b['length'] * 1e3:.0f}, IoU hanging {b['iou_hanging']:.2f} swinging {b['iou_swing']:.2f}, "
                  f"period {b['period']:.3f} s (model {b['period_model']:.3f})", flush=True)
        else:
            print(f"{p}: not measured: {o['reason']}", flush=True)


if __name__ == "__main__":
    main()
