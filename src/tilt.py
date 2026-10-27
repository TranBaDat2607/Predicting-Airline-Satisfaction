"""Public-label tilt: re-rank a base file toward an estimate of which test rows are public positives/negatives,
using nothing but files with a known (rounded) public AUC.

Model. Row i is a public positive (a_i = 1), a public negative (b_i = 1) or private. Prior: public with
probability f, positive with the base's calibrated probability p_i. For file k with ranking s_k, the public AUC
difference to the base b is exactly sum_{i,j} a_i b_j [H_k(i,j) - H_b(i,j)] / (n1 n0) (H = "i ranked above j").
Linearised around the prior means this is linear in (a, b):

    n1 n0 (A_k - A_b) - c_k = sum_i (a_i - f p_i) g_k(i) + (b_i - f (1 - p_i)) h_k(i) + noise
    g_k(i) = f [W_k(i) - W_b(i)],  W(i) = sum of (1 - p_j) over rows ranked below i   (negatives i passes)
    h_k(i) = f [U_k(i) - U_b(i)],  U(i) = sum of p_j over rows ranked above i         (positives above i)

A Gaussian prior on (a, b) gives the posterior mean in closed form (a K x K solve). The tilted score is
P(positive | public) = E[a] / (E[a] + E[b]), with the update scaled by eta (eta = 0 returns the base).
Noise = the 5-decimal rounding of every score (the base's is shared by all files), inflated by noise_units.

This is public-LB fitting: it raises the public score and costs private score (S6E9: about -0.9u private per
+1u public). Files built with it are tagged [public-LB fitted] and are never final picks.
"""
from __future__ import annotations

import numpy as np


class Tilt:
    def __init__(self, p: np.ndarray, f: float = 0.2, noise_units: float = 1.0, nuisance: bool = True):
        self.p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        self.f = f
        self.nuisance = nuisance
        self.n = len(self.p)
        self.abar = f * self.p
        self.bbar = f * (1 - self.p)
        self.va = f * self.p * (1 - f * self.p)
        self.vb = f * (1 - self.p) * (1 - f * (1 - self.p))
        self.cab = -f * f * self.p * (1 - self.p)
        self.n1 = f * self.p.sum()
        self.n0 = f * (1 - self.p).sum()
        self.sd = noise_units * 1e-5 * self.n1 * self.n0 / np.sqrt(12.0)
        self.wb, self.ub = self._wu(self.p)
        self.sb = float(self.p @ self.wb)

    def _wu(self, s):
        order = np.argsort(np.asarray(s, dtype=np.float64), kind="stable")
        q = 1 - self.p[order]
        pp = self.p[order]
        w = np.empty(self.n)
        u = np.empty(self.n)
        w[order] = np.cumsum(q) - q
        u[order] = pp.sum() - np.cumsum(pp)
        return w, u

    def fit(self, files, scores, base_score):
        self.base_score = base_score
        self.g = np.empty((0, self.n), dtype=np.float32)
        self.h = np.empty((0, self.n), dtype=np.float32)
        self.yraw, self.cs, self.dist = np.empty(0), np.empty(0), np.empty(0)
        self.m = np.empty((0, 0))
        return self.add(files, scores)

    def add(self, files, scores):
        """Append scored files (incremental: only the new rows and columns of the K x K system are computed)."""
        k = len(files)
        g = np.empty((k, self.n), dtype=np.float32)
        h = np.empty((k, self.n), dtype=np.float32)
        yv, cs, dist = np.empty(k), np.empty(k), np.empty(k)
        for i, s in enumerate(files):
            w, u = self._wu(s)
            g[i] = self.f * (w - self.wb)
            h[i] = self.f * (u - self.ub)
            cs[i] = self.f ** 2 * (float(self.p @ w) - self.sb)
            dist[i] = float(np.square(w - self.wb).mean())
            yv[i] = self.n1 * self.n0 * (scores[i] - self.base_score) - cs[i]
        va, vb, cab = (x.astype(np.float32) for x in (self.va, self.vb, self.cab))
        gv, hv, gc, hc = g * va, h * vb, g * cab, h * cab
        new = (gv @ g.T + hv @ h.T + gc @ h.T + hc @ g.T).astype(np.float64)
        cross = (gv @ self.g.T + hv @ self.h.T + gc @ self.h.T + hc @ self.g.T).astype(np.float64)
        self.m = np.block([[self.m, cross.T], [cross, new]])
        self.g, self.h = np.concatenate([self.g, g]), np.concatenate([self.h, h])
        self.yraw, self.cs, self.dist = (np.concatenate(x) for x in ((self.yraw, yv), (self.cs, cs), (self.dist, dist)))
        return self.solve()

    def solve(self):
        g, h, yv, cs, dist, m = self.g, self.h, self.yraw.copy(), self.cs, self.dist, self.m
        k = len(yv)
        e = self.sd ** 2 * (np.eye(k) + np.ones((k, k)))
        q = np.linalg.inv(m + e)
        if self.nuisance:
            # calibration errors of the prior scale the expected loss c_k and grow with the distance to the
            # base: remove both directions (flat-prior GLS) so only consistent row-level evidence drives the tilt
            x = np.stack([cs / np.abs(cs).max(), dist / max(dist.max(), 1e-12)], 1)
            beta = np.linalg.solve(x.T @ q @ x, x.T @ q @ yv)
            yv = yv - x @ beta
            self.beta = beta
        self.lam = q @ yv
        # closed-form leave-one-out: how well the other files predict each file's public score surprise
        self.loo = self.lam / np.diag(q)
        self.loo_r2 = 1 - float(self.loo @ self.loo) / float(yv @ yv)
        gl = g.T.astype(np.float64) @ self.lam
        hl = h.T.astype(np.float64) @ self.lam
        self.da = self.va * gl + self.cab * hl
        self.db = self.cab * gl + self.vb * hl
        self.y = yv
        return self

    def predict(self, s):
        """Posterior-mean public score of a candidate ranking s (plus the nuisance part of the fit)."""
        w, u = self._wu(s)
        g = self.f * (w - self.wb)
        h = self.f * (u - self.ub)
        c = self.f ** 2 * (float(self.p @ w) - self.sb)
        y = float(self.da @ g + self.db @ h)
        if self.nuisance:
            d = float(np.square(w - self.wb).mean())
            y += self.beta[0] * c / np.abs(self.cs).max() + self.beta[1] * d / max(self.dist.max(), 1e-12)
        return self.base_score + (y + c) / (self.n1 * self.n0)

    def score(self, eta: float = 1.0):
        a = np.clip(self.abar + eta * self.da, 1e-9, None)
        b = np.clip(self.bbar + eta * self.db, 1e-9, None)
        return a / (a + b)
