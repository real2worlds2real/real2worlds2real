"""Derivative-free optimizers over the unit box [0, 1]^d with batched ask / tell: one generation = one batch of
scene copies (apps/real2sim_replay.py --batch). Out-of-box samples are evaluated at their clipped point plus a quadratic
penalty, so the search distribution learns to stay inside.
"""
import numpy as np


def penalised(f_batch, X, penalty=1e3):
    Xc = np.clip(X, 0.0, 1.0)
    return np.asarray(f_batch(Xc), float) + penalty * ((X - Xc) ** 2).sum(1)


class CMAES:
    """(mu/mu_w, lambda)-CMA-ES (Hansen's tutorial defaults) with rank-one and rank-mu covariance updates."""

    def __init__(self, x0, sigma=0.25, popsize=None, seed=0):
        self.m = np.asarray(x0, float).copy()
        n = self.n = len(self.m)
        self.sigma = sigma
        self.lam = popsize or 4 + int(3 * np.log(n))
        mu = self.lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.pc, self.ps, self.C = np.zeros(n), np.zeros(n), np.eye(n)
        self.rng = np.random.default_rng(seed)
        self.gen, self.best = 0, (None, np.inf)

    def ask(self):
        self.D2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(self.D2, 1e-20))
        self.Z = self.rng.standard_normal((self.lam, self.n))
        self.Y = self.Z @ np.diag(self.D) @ self.B.T
        return self.m + self.sigma * self.Y

    def tell(self, X, f):
        f = np.asarray(f, float)
        order = np.argsort(f)
        if f[order[0]] < self.best[1]:
            self.best = (np.clip(X[order[0]], 0, 1).copy(), float(f[order[0]]))
        mu = len(self.w)
        Ysel = self.Y[order[:mu]]
        yw = self.w @ Ysel
        self.m = self.m + self.sigma * yw
        Cinvsqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cinvsqrt @ yw
        self.gen += 1
        ps_norm = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen))
        hs = ps_norm < (1.4 + 2 / (self.n + 1)) * self.chin
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (Ysel.T * self.w) @ Ysel
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc)
                  + (1 - hs) * self.cc * (2 - self.cc) * self.C) + self.cmu * rank_mu)
        self.C = (self.C + self.C.T) / 2
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))
        self.sigma = float(min(self.sigma, 1.0))

    def done(self, tol=1e-4):
        return self.sigma * np.sqrt(np.max(np.diag(self.C))) < tol


class CEM:
    """Cross-entropy method: diagonal Gaussian refit to the elite fraction, with smoothing and a variance floor."""

    def __init__(self, x0, sigma=0.3, popsize=32, elite=0.2, alpha=0.7, min_std=1e-3, seed=0):
        self.m = np.asarray(x0, float).copy()
        self.s = np.full(len(self.m), sigma)
        self.lam, self.k, self.alpha, self.min_std = popsize, max(2, int(elite * popsize)), alpha, min_std
        self.rng = np.random.default_rng(seed)
        self.best = (None, np.inf)

    def ask(self):
        return self.m + self.s * self.rng.standard_normal((self.lam, len(self.m)))

    def tell(self, X, f):
        f = np.asarray(f, float)
        order = np.argsort(f)
        if f[order[0]] < self.best[1]:
            self.best = (np.clip(X[order[0]], 0, 1).copy(), float(f[order[0]]))
        E = np.clip(X[order[:self.k]], 0, 1)
        self.m = self.alpha * E.mean(0) + (1 - self.alpha) * self.m
        self.s = np.maximum(self.alpha * E.std(0) + (1 - self.alpha) * self.s, self.min_std)

    def done(self, tol=1e-4):
        return float(self.s.max()) < tol


def minimize(f_batch, opt, iters=50, log=None):
    """Run `opt` on f_batch((n, d) unit-box points) -> (n,) losses; returns (best x, best f, history of best f)."""
    hist = []
    for g in range(iters):
        X = opt.ask()
        opt.tell(X, penalised(f_batch, X))
        hist.append(opt.best[1])
        if log:
            log(g, opt.best[1])
        if opt.done():
            break
    return opt.best[0], opt.best[1], hist
