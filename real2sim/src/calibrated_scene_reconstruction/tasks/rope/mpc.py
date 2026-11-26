"""Model predictive control of the rope letters' pick-and-places (apps/newton_rope_letter.py --planner mpc): before
each pick-and-place, candidates are rolled out in a model of the rope and the one whose rope ends closest to the
letter, looking two pick-and-places ahead, is executed; the rope is then observed again and the next one planned
(receding horizon, one pick-and-place a step).

A candidate takes one segment and moves it along a path: either straight to a point (where it is set down) or, a
trace, to a point on the letter and then dragged along the letter's path to the segment's own place, the segments
behind it following the path and laid along a whole stroke (one point moved at a time cannot shift a stroke lying a
few cm off its line).

The model is the rope's segment centres in the table's plane, dragged as a chain on a high-friction table: the taken
segment follows the hand's path, and a neighbour moves only when the chain pulls it taut (follow the leader), never
pushed (a rope heavy enough for the table's friction to hold its bends: apps/newton_rope_letter.py's --density).
What the model gets wrong (a pinch that did not hold, a pick-and-place that left the rope farther from the letter) is
not chosen again. Stdlib and numpy only (it runs in the Newton image's python).
"""
from __future__ import annotations

import numpy as np


def errors(P, G):
    """Each candidate's per-segment distances from the letter, the goal matched end to end the closer way round.
    P: (..., n, 2), G: (n, 2)."""
    e1 = np.linalg.norm(P - G, axis=-1)
    e2 = np.linalg.norm(P - G[::-1], axis=-1)
    return np.where((e1.mean(-1) <= e2.mean(-1))[..., None], e1, e2)


def cost(P, G, tol):
    """Lower is closer to done: the mean and the largest distance, a segment in place counting as 0."""
    e = np.maximum(errors(P, G) - 0.5 * tol, 0.0)
    return e.mean(-1) + 0.5 * e.max(-1)


def carry(P, k, path, link, step=0.004, max_steps=120):
    """The rope after each candidate: P (C, n, 2) segment centres, k (C,) the segment taken, path (C, W, 2) the points
    it is moved through from where it lies (the last where it is set down), link the distance between neighbouring
    centres."""
    P = P.copy()
    C, n, _ = P.shape
    c = np.arange(C)
    full = np.concatenate([P[c, k][:, None], path], 1)
    seg = np.linalg.norm(np.diff(full, axis=1), axis=2)
    cum = np.concatenate([np.zeros((C, 1)), np.cumsum(seg, 1)], 1)
    total = cum[:, -1]
    steps = int(np.clip(np.ceil(total.max() / step), 2, max_steps))
    d = np.arange(1, n)
    for s in range(1, steps + 1):
        u = total * (s / steps)
        i = np.clip((cum[:, 1:] < u[:, None]).sum(1), 0, path.shape[1] - 1)
        f = np.clip((u - cum[c, i]) / np.maximum(seg[c, i], 1e-9), 0.0, 1.0)
        P[c, k] = full[c, i] + f[:, None] * (full[c, i + 1] - full[c, i])
        for side in (1, -1):
            for dd in d:
                j = k + side * dd
                ok = (j >= 0) & (j < n)
                if not ok.any():
                    break
                cc, jj = c[ok], j[ok]
                prev = P[cc, jj - side]
                v = P[cc, jj] - prev
                r = np.linalg.norm(v, axis=1)
                far = r > link  # pulled taut by its neighbour nearer the hand
                P[cc[far], jj[far]] = prev[far] + v[far] * (link / r[far])[:, None]
    return P


def candidates(X, G, err, tol, rng, n_noise=6, noise=0.01, traces=(3, 6, 10)):
    """(k, path) pairs: every segment out of place set down at its place, overshooting it along the way it comes (to
    make up for the chain dragging it back) or near it; and every segment traced into place over the m segments of
    the letter on either side of it (carried to the place of the segment m along, dragged back along the letter)."""
    n = len(X)
    ks, paths = [], []
    for k in np.flatnonzero(err > 0.5 * tol):
        g, x = G[k], X[k]
        for a in (0.0, 0.15, 0.3):
            ks.append(k)
            paths.append([g + a * (g - x)])
        for t in g + rng.normal(0.0, noise, (n_noise, 2)):
            ks.append(k)
            paths.append([t])
    for k in range(n):
        for side in (1, -1):
            for m in traces:
                j = k + side * m
                if 0 <= j < n and (err[min(k, j): max(k, j) + 1] > 0.5 * tol).any():
                    ks.append(k)
                    paths.append(G[j:k - side if k - side >= 0 else None:-side])
    w = max(len(p) for p in paths)
    padded = np.array([np.concatenate([p, np.repeat(np.asarray(p)[-1:], w - len(p), 0)]) for p in paths], float)
    return np.asarray(ks, int), padded, [np.asarray(p, float) for p in paths]


def matched(X, G):
    e1 = np.linalg.norm(X - G, axis=1)
    e2 = np.linalg.norm(X - G[::-1], axis=1)
    return (G, e1) if e1.mean() <= e2.mean() else (G[::-1], e2)


class RopeMPC:
    """plan(X) -> (segment, (x, y) it ends at, info with the path when a trace), from the rope's segment centres X."""

    def __init__(self, goal, link, tol, seed=0, beam=12):
        self.goal, self.link, self.tol, self.beam = np.asarray(goal, float), link, tol, beam
        self.rng = np.random.default_rng(seed)
        self.avoid = []  # (where the pinch was, where it ended or None): what the model got wrong, not chosen again

    def went_wrong(self, pick, end=None):
        """A pick-and-place the model got wrong: the pinch at pick did not hold (end None: a tight bend, two strands
        under the fingers), or the rope ended farther from the letter than before (end: where it was set down). The
        model would choose it again and again from the same rope; it is not chosen again there."""
        self.avoid.append((np.asarray(pick, float)[:2], None if end is None else np.asarray(end, float)[:2]))

    def rollout(self, X, k, path):
        return carry(np.repeat(X[None], len(k), 0), k, path, self.link)

    def best_next(self, P):
        """The lowest cost one greedy pick-and-place (each segment set down at its place) leaves each rope at."""
        out = []
        for X in P:
            G, err = matched(X, self.goal)
            if (err < self.tol).all():
                out.append(0.0)
                continue
            ks = np.flatnonzero(err > 0.5 * self.tol)
            Q = self.rollout(X, ks, G[ks][:, None])
            out.append(float(cost(Q, self.goal, self.tol).min()))
        return np.asarray(out)

    def plan(self, X):
        X = np.asarray(X, float)[:, :2]
        G, err = matched(X, self.goal)
        ks, padded, paths = candidates(X, G, err, self.tol, self.rng)
        ok = np.ones(len(ks), bool)
        for pick, end in self.avoid:  # whichever segment lies there now
            near = np.linalg.norm(X[ks] - pick, axis=1) < 0.015
            if end is not None:
                near &= np.linalg.norm(padded[:, -1] - end, axis=1) < 0.015
            ok &= ~near
        if ok.any():
            ks, padded, paths = ks[ok], padded[ok], [p for p, o in zip(paths, ok) if o]
        P = self.rollout(X, ks, padded)
        J1 = cost(P, self.goal, self.tol)
        top = np.argsort(J1)[: self.beam]
        J2 = self.best_next(P[top])
        i = top[int(np.argmin(np.minimum(J1[top], J2) + 0.5 * J2))]  # the rope one step on, and where it can go next
        info = {"predicted_cost": float(J1[i]), "predicted_errors": errors(P[i], self.goal).round(4).tolist(),
                "oracle_segment": int(np.argmax(err)), "candidates": int(len(ks))}
        if len(paths[i]) > 1:
            info["path"] = paths[i].round(4).tolist()
        return int(ks[i]), paths[i][-1], info
