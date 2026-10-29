"""Empirical-Bayes noise model for the public-label tilt (src/tilt.py).

tilt.py gives every scored file the same noise sd (one rounding step x noise_units). Files differ: near-base files
(own LB-blend probes, atlas probes) are predicted by the other files to < 1u, far notebook files carry a
linearisation error that grows with their distance to the base, and some notebook scores belong to another
version of the notebook than the output we downloaded (gross errors). Here the per-file noise variance is

    var_k = r2 * (k0 + k1 * M_kk / median(M_kk)) / w_k

r2 = variance of one 5-decimal rounding, M_kk = prior variance of file k's public-score surprise (its size),
w_k = Student-t weight (EM; nu degrees of freedom) that down-weights files the others cannot explain.
(k0, k1) maximise the REML marginal likelihood of the surprises; the base's rounding is shared by all files.

    from tilt_eb import TiltEB
    t = TiltEB(p).fit(files, scores, base_score)      # builds g, h, M once
    t.fit_noise(nu=4)                                 # REML (k0, k1) + Student-t weights; then t.score(eta)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from tilt import Tilt


class TiltEB(Tilt):
    def __init__(self, p, f=0.2, nuisance=True):
        super().__init__(p, f=f, noise_units=1.0, nuisance=nuisance)
        self.r2 = self.sd ** 2  # variance of one rounding step in pair units (sd already / sqrt 12)
        self.noise_var = None
        self.w = None

    # --- linear algebra for a given noise vector ---------------------------------------------------------------
    def _cov(self, nv):
        k = len(self.yraw)
        return self.m + np.diag(nv) + self.r2 * np.ones((k, k))

    def _x(self):
        cs, dist = self.cs, self.dist
        return np.stack([cs / np.abs(cs).max(), dist / max(dist.max(), 1e-12)], 1)

    def reml(self, nv):
        c = self._cov(nv)
        try:
            ch = np.linalg.cholesky(c)
        except np.linalg.LinAlgError:
            return -np.inf
        ci = np.linalg.inv(c)
        y = self.yraw
        if self.nuisance:
            x = self._x()
            a = x.T @ ci @ x
            beta = np.linalg.solve(a, x.T @ ci @ y)
            r = y - x @ beta
            ld_a = np.linalg.slogdet(a)[1]
        else:
            r, ld_a = y, 0.0
        return -0.5 * (2 * np.log(np.diag(ch)).sum() + ld_a + r @ ci @ r)

    def noise_vector(self, k0, k1, w=None):
        mk = np.diag(self.m)
        nv = self.r2 * (k0 + k1 * mk / np.median(mk))
        if w is not None:
            nv = nv / w
        return nv

    def fit_noise(self, nu: float = 0.0, iters: int = 30, fixed=None, exact=None, verbose=True):
        """REML over (log k0, log k1); with nu > 0, Student-t EM weights on top (re-optimising k0, k1 each pass).
        exact: boolean mask of files whose scores are exact (own submissions): their weight is never below 1."""
        k = len(self.yraw)
        w = np.ones(k)
        exact = np.zeros(k, bool) if exact is None else np.asarray(exact, bool)

        def negll(th, w):
            return -self.reml(self.noise_vector(np.exp(th[0]), np.exp(th[1]), w))

        th = np.log([1.0, 1.0]) if fixed is None else np.log(fixed)
        for it in range(iters if nu > 0 else 1):
            if fixed is None:
                res = minimize(negll, th, args=(w,), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3})
                th = res.x
            nv = self.noise_vector(np.exp(th[0]), np.exp(th[1]), w)
            if nu <= 0:
                break
            # E-step: posterior of the noise of each file given all data, e_k ~ N(m_k, v_k)
            c = self._cov(nv)
            ci = np.linalg.inv(c)
            r = self._resid(ci)
            mvec = nv * (ci @ r)
            vvec = nv - nv * np.diag(ci) * nv
            base = self.noise_vector(np.exp(th[0]), np.exp(th[1]))
            w_new = (nu + 1) / (nu + (mvec ** 2 + vvec) / base)
            w_new[exact] = np.maximum(w_new[exact], 1.0)
            if np.abs(np.log(w_new / w)).max() < 1e-3:
                w = w_new
                break
            w = w_new
        self.k0, self.k1 = np.exp(th)
        self.w = w
        self.noise_var = self.noise_vector(self.k0, self.k1, w)
        self.solve_nv(self.noise_var)
        if verbose:
            print(f"REML noise: k0 {self.k0:.3g} k1 {self.k1:.3g} (x rounding var); "
                  f"weights < 0.5: {(w < 0.5).sum()}, < 0.2: {(w < 0.2).sum()}; REML {self.reml(self.noise_var):.2f}")
        return self

    def _resid(self, ci):
        y = self.yraw
        if not self.nuisance:
            self.beta = np.zeros(2)
            return y
        x = self._x()
        self.beta = np.linalg.solve(x.T @ ci @ x, x.T @ ci @ y)
        return y - x @ self.beta

    def solve_nv(self, nv):
        c = self._cov(nv)
        q = np.linalg.inv(c)
        yv = self._resid(q)
        self.lam = q @ yv
        self.loo = self.lam / np.diag(q)
        self.loo_r2 = 1 - float(self.loo @ self.loo) / float(yv @ yv)
        gl = self.g.T.astype(np.float64) @ self.lam
        hl = self.h.T.astype(np.float64) @ self.lam
        self.da = self.va * gl + self.cab * hl
        self.db = self.cab * gl + self.vb * hl
        self.y = yv
        return self

    def solve(self):  # keep the parent's incremental add() working with the current noise model
        if self.noise_var is None or len(self.noise_var) != len(self.yraw):
            return super().solve()
        return self.solve_nv(self.noise_var)
